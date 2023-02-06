//----------------------------//
// This file is part of RaiSim//
// Copyright 2020, RaiSim Tech//
//----------------------------//

#pragma once

#include <cmath>
#include <stdlib.h>
#include <set>
#include "../../RaisimGymEnv.hpp"
#include <vector>

namespace raisim {

class ENVIRONMENT : public RaisimGymEnv {

 public:

  explicit ENVIRONMENT(const std::string& resourceDir, const Yaml::Node& cfg, bool visualizable) :
      RaisimGymEnv(resourceDir, cfg), visualizable_(visualizable), normDist_(0, 1) {


    raisim::World::setActivationKey("~/.raisim"); //path of the folder in which i have the activation key
    /// create world
    world_ = std::make_unique<raisim::World>();

    /// add objects
    //anymal_ = world_->addArticulatedSystem(resourceDir_+"/anymal/urdf/anymal.urdf");
    anymal_ = world_->addArticulatedSystem("/home/claudio/raisim_ws/raisimlib/rsc/anymal/urdf/anymal.urdf");
    anymal_->setName("anymal");
    anymal_->setControlMode(raisim::ControlMode::PD_PLUS_FEEDFORWARD_TORQUE);
    world_->addGround();

    READ_YAML(double, num_steps, cfg_["num_steps"]);
    READ_YAML(double, width_step, cfg_["width_step"]);
    READ_YAML(double, height_step, cfg_["height_step"]);
    READ_YAML(double, curriculumFactor_, cfg["curriculum"]["curriculumFactor_"])
    READ_YAML(double, curriculumDecayFactor_, cfg["curriculum"]["curriculumDecayFactor_"])
  
    //using t_value = typename std::iterator_traits<Iter>::value_type;
    //std::vector<t_value> temp(size);

    ///Lo metto pure nel costruttore per fissare la dimensione del vector 'boxes', poi il suo contenuto verra' aggiornato nel curriculmum update
    for (int i=0; i<num_steps; i++){
      boxes.push_back(world_->addBox(width_step -1, width_step-1, height_step, 10));
      boxes[i]->setName("step" + std::to_string(i)+"-th");
      boxes[i]->setPosition(25,0,0.1*i);
      steps_indices.push_back(boxes[i]->getIndexInWorld()); //riempe il vettore degli indici dei box (dei semplici numeri che indicano gli oggetti nel world)
    }

    /// get robot data
    READ_YAML(int, num_seq, cfg_["num_seq"]);

    gcDim_ = anymal_->getGeneralizedCoordinateDim();
    gvDim_ = anymal_->getDOF();
    nJoints_ = gvDim_ - 6;
    /// initialize containers
    current_action_.setZero(12);
    previous_action_.setZero(12);
    previous_gv_.setZero(12);

    joint_history_pos_.setZero(num_legs_joint*num_seq);
    joint_history_vel_.setZero(num_legs_joint*num_seq);

    joint_container_.setZero(24);
    joint_history_.setZero(joint_container_.size()*num_seq); //3 timestep to save the joint history.
    q1.setZero(num_seq), q2.setZero(num_seq), q3.setZero(num_seq), q4.setZero(num_seq), q5.setZero(num_seq), q6.setZero(num_seq);
    q7.setZero(num_seq), q8.setZero(num_seq), q9.setZero(num_seq), q10.setZero(num_seq), q11.setZero(num_seq), q12.setZero(num_seq);

    standing_configuration_.setZero(12);
    actual_joint_position_.setZero(12);
    actual_joint_velocities_.setZero(12);
    command_<< 1, 0.2, 0;
    
    gc_.setZero(gcDim_); gc_init_.setZero(gcDim_);
    gv_.setZero(gvDim_); gv_init_.setZero(gvDim_);
    ga_.setZero(nJoints_); ga_init_.setZero(nJoints_);

    pTarget_.setZero(gcDim_); vTarget_.setZero(gvDim_); pTarget12_.setZero(nJoints_);

    /// this is nominal configuration of anymal
    gc_init_ << 0, 0, 0.50, 1.0, 0.0, 0.0, 0.0, 0.03, 0.4, -0.8, -0.03, 0.4, -0.8, 0.03, -0.4, 0.8, -0.03, -0.4, 0.8;
    standing_configuration_ = gc_init_.tail(12);

    /// set pd gains
    Eigen::VectorXd jointPgain(gvDim_), jointDgain(gvDim_);
    jointPgain.setZero(); jointPgain.tail(nJoints_).setConstant(50.0);
    jointDgain.setZero(); jointDgain.tail(nJoints_).setConstant(0.2);
    anymal_->setPdGains(jointPgain, jointDgain);
    anymal_->setGeneralizedForce(Eigen::VectorXd::Zero(gvDim_));

    /// MUST BE DONE FOR ALL ENVIRONMENTS
   

    if(&cfg["action_in_observation_space"]){ //because we don't use the READ_YAML we don't need underscore
      obDim_ = 34 + nJoints_;
      action_in_observation_space_ = true;
      if(&cfg["joint_history_in_observation_space"]){
        obDim_ = obDim_ + joint_history_.size();
        joint_history_in_observation_space_ = true;
      }
    }else
      obDim_ = 34;

    actionDim_ = nJoints_; actionMean_.setZero(actionDim_); actionStd_.setZero(actionDim_);
    obDouble_.setZero(obDim_);

    /// action scaling
    actionMean_ = gc_init_.tail(nJoints_);
    double action_std;
    READ_YAML(double, action_std, cfg_["action_std"]) /// example of reading params from the config. Use the element with _ in READ_YAML otherwise use it without underscore
    actionStd_.setConstant(action_std);

    /// Reward coefficients
    rewards_.initializeFromConfigurationFile (cfg["reward"]);
    //std::cout<<"Reward from yaml: "<<cfg["reward"]<<std::endl;

    /// indices of links that should not make contact with ground
    footIndices_.insert(anymal_->getBodyIdx("LF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RF_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("LH_SHANK"));
    footIndices_.insert(anymal_->getBodyIdx("RH_SHANK"));
    baseIndex_ = anymal_->getBodyIdx("base");

    footLinkFrame_.insert(anymal_->getFrameIdxByLinkName("LF_SHANK"));  //l'indice del frame e' sempre un numero che ci dice se quel frame e' riferito a qualcosa o no
    footLinkFrame_.insert(anymal_->getFrameIdxByLinkName("RF_SHANK"));
    footLinkFrame_.insert(anymal_->getFrameIdxByLinkName("LH_SHANK"));
    footLinkFrame_.insert(anymal_->getFrameIdxByLinkName("RH_SHANK"));

    /// visualize if it is the first environment
   if (visualizable_) {
      server_ = std::make_unique<raisim::RaisimServer>(world_.get());
      server_->launchServer();
      server_->focusOn(anymal_);
    }
  }

  void init() final { }

  void reset() final {
    anymal_->setState(gc_init_, gv_init_);
    updateObservation(); //when we reset, we don't mind of the previous action
  }


  float step(const Eigen::Ref<EigenVec>& action) final {
    /// action scaling
    pTarget12_ = action.cast<double>();
    current_action_ = pTarget12_;  //I put the last action in the observation space

    pTarget12_ = pTarget12_.cwiseProduct(actionStd_);
    pTarget12_ += actionMean_;
    pTarget_.tail(nJoints_) = pTarget12_;

    anymal_->setPdTarget(pTarget_, vTarget_);

    for(int i=0; i< int(control_dt_ / simulation_dt_ + 1e-10); i++){
      if(server_) server_->lockVisualizationServerMutex();
      world_->integrate();
      if(server_) server_->unlockVisualizationServerMutex();
    }
    
    updateJointHistory(); 
    if(action_in_observation_space_)
      updateObservation(current_action_);
    else
      updateObservation();

    hipPenalty();
    footSlipage(); //Check the value of c_F during the action

    if(isFellDownTheStairs())
      number_of_falls_ +=1;

    rewards_.record("torque", anymal_->getGeneralizedForce().squaredNorm());
    rewards_.record("forwardVel", std::exp(- std::pow((command_[0] - bodyLinearVel_[0]),2)/0.25));
    //rewards_.record("forwardVel", std::exp(- std::pow((command_[1] - bodyLinearVel_[1]),2)/0.25));
    rewards_.record("angularVel", std::exp(- std::pow((command_[2] - bodyAngularVel_[2]),2)/0.25));

    rewards_.record("Joint_velocity", gv_.tail(12).norm());
    rewards_.record("Smooth_action", (current_action_ - previous_action_).norm()); //penalize big distance between 2 actions
    rewards_.record("number_of_contact", numberOfContact_);
    rewards_.record("omega_x", std::pow(bodyAngularVel_[0], 2));
    rewards_.record("omega_y", std::pow(bodyAngularVel_[1], 2));
    rewards_.record("v_z", std::pow(bodyLinearVel_[2], 2));
    rewards_.record("hip_penalty", hipTerm_); //The more the error is bigger, the more penalize this term
    //rewards_.record("gait_term_pos", gaitTerm_Pos_);
    //rewards_.record("gait_term_vel", gaitTerm_Vel_);
    rewards_.record("thigh_penalty", thighTerm_);
    rewards_.record("slippage", slip_term_);

    previous_action_ = current_action_;  //Save the current action to compute the next reward tem
    return rewards_.sum();
  }


  

  void updateObservation(Eigen::VectorXd current_action_) {
    
    if(anymal_)
      anymal_->getState(gc_, gv_);  //Update the value of the joints. 

    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);

    if(joint_history_in_observation_space_){
      obDouble_ << gc_[2], /// body height
        rot.e().row(2).transpose(), /// body orientation
        bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity  
        current_action_, //last action
        joint_history_pos_,
        joint_history_vel_,
        command_;
    }else
      obDouble_ << gc_[2], /// body height
          rot.e().row(2).transpose(), /// body orientation
          gc_.tail(12), /// joint angles
          bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
          gv_.tail(12), /// joint velocity
          current_action_;

    joint_container_ << gc_.tail(12), gv_.tail(12); //over write the content: 

    actual_joint_position_ = joint_container_(Eigen::seq(0,11));
    actual_joint_velocities_ = joint_container_(Eigen::seq(12,23));

    previous_gv_ = gv_.tail(12);
  }


  void updateObservation() {

    anymal_->getState(gc_, gv_);  //Update the value of the joints. 

    raisim::Vec<4> quat;
    raisim::Mat<3,3> rot;
    quat[0] = gc_[3]; quat[1] = gc_[4]; quat[2] = gc_[5]; quat[3] = gc_[6];
    raisim::quatToRotMat(quat, rot);
    bodyLinearVel_ = rot.e().transpose() * gv_.segment(0, 3);
    bodyAngularVel_ = rot.e().transpose() * gv_.segment(3, 3);
  
    obDouble_ << gc_[2], /// body height
          rot.e().row(2).transpose(), /// body orientation
          gc_.tail(12), /// joint angles
          bodyLinearVel_, bodyAngularVel_, /// body linear&angular velocity
          gv_.tail(12); /// joint velocity
  }

  void updateJointHistory(){
      
    joint_history_pos_.tail(12) = gc_.tail(12);
    joint_history_vel_.tail(12) = gv_.tail(12);

    int i=num_seq-2;
    while(i !=0){
      joint_history_pos_(Eigen::seq(i*num_legs_joint, (i+1)*num_legs_joint-1)) = joint_history_pos_(Eigen::seq((i+1)*num_legs_joint, (i+2)*num_legs_joint-1));
      joint_history_vel_(Eigen::seq(i*num_legs_joint, (i+1)*num_legs_joint-1)) = joint_history_vel_(Eigen::seq((i+1)*num_legs_joint, (i+2)*num_legs_joint-1));
      i--;
    }

    for(int i=0; i<num_seq-1; i++){
      q1<<joint_history_pos_[i*12];
      q2<<joint_history_pos_[i*12+1];
      q3<<joint_history_pos_[i*12+2];
      q4<<joint_history_pos_[i*12+3];
      q5<<joint_history_pos_[i*12+4];
      q6<<joint_history_pos_[i*12+5];
      q7<<joint_history_pos_[i*12+6];
      q8<<joint_history_pos_[i*12+7];
      q9<<joint_history_pos_[i*12+8];
      q10<<joint_history_pos_[i*12+9];
      q11<<joint_history_pos_[i*12+10];
      q12<<joint_history_pos_[i*12+11];
    }
  }
  

  void hipPenalty(){
    hipTerm_ = 0.0; //Altrimenti episodio dopo episodio sto termine cresce sempre.
    thighTerm_ = 0.0;

    for(int i=0; i<4; i++){
      hipTerm_ += std::pow(actual_joint_position_(i*3) - standing_configuration_(i*3), 2); //il +1 e' necessario perche' il primo giunto e' quello della base
      thighTerm_ += std::pow(actual_joint_position_(i*3 + 1) - standing_configuration_(i*3 + 1), 2);
    }
    /*Gli indici dei giunti di hip perche' nella cartella raisimExample ho usato il metodo che mi ritorna il nome dei giunti. Il file e' Anymal_numberOfJoints.cpp */
  }

  void generate_command_velocity(){ 
    std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<> dis(0, 1.4);
    for (int n = 0; n < 3; n++) {
        // Use dis to transform the random unsigned int generated by gen into a 
        // double in [1, 2). Each call to dis(gen) generates a new random double
        command_[n] = dis(gen);
    }
  }


 
  void footSlipage(){

    closedContact();
    int i = 0;
    slip_term_ = 0.0;
    std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> footVelocity;
    //std::cout<<"indirizzo di memoria: "<< &footVelocity<<std::endl;
    //std::cout<<"indirizzo di memoria: "<< &tan_vel_norm<<std::endl;

    for(std::set<size_t>::iterator it=footLinkFrame_.begin(); it!=footLinkFrame_.end(); ++it){
      anymal_->getFrameVelocity(*it, temp_footVelocity);
      footVelocity.push_back(temp_footVelocity.e()); 
    }

    for(std::vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>>::iterator it=footVelocity.begin(); it!=footVelocity.end(); ++it, i++){
      tan_vel_norm = (*it)(Eigen::seq(0,1)).norm(); //(*it) mi da il contenuto di footVelocity, che e' un Eigen vectore, con seq prendo i primi 2 elementi, e poi ne faccio la norma
      if(cF_[i]==1)
        slip_term_ += tan_vel_norm;
      //tan_vel_norm_vec.push_back(tan_vel_norm);
    }
    
    //slip_term_ = std::inner_product(tan_vel_norm_vec.begin(), tan_vel_norm_vec.end(), cF_.begin(), 0);
  }


 void closedContact(){
    //closed contatct means that the foot is on the ground (stance phase)
    int i=0;
    for(std::set<size_t>::iterator it=footIndices_.begin(); it!=footIndices_.end(); ++it, i++){
      for(auto& contact: anymal_->getContacts()){
        if(contact.getlocalBodyIndex() == *it){ //se il piede tocca due oggetti contemporaneamente lo stesso indice di contatto comparira' 2 volte
          cF_[i]=1;
        }
        else
          cF_[i]=0;
      }
    }

    //std::cout<<"dim contact vector: "<< cF_.size()<<std::endl;
  }
  




  void observe(Eigen::Ref<EigenVec> ob) final {
    /// convert it to float
    ob = obDouble_.cast<float>();
  }

  bool isTerminalState(float& terminalReward) final {
    terminalReward = float(terminalRewardCoeff_);

    /// if the contact body is not feet
    for(auto& contact: anymal_->getContacts())
      if(footIndices_.find(contact.getlocalBodyIndex()) == footIndices_.end()){ //e' come fare un if(indice contatto != footIndeces_[i]) allora il contatto non e' ai piedi
        numberOfContact_ += numberOfContact_;
        return true;
      }

    terminalReward = 0.f;
    return false;
  }

  bool isFellDownTheStairs(){ 

    for(auto& contact: anymal_->getContacts()){
      
      if(contact.getlocalBodyIndex() == baseIndex_){ //il contatto e' alla base
        std::cout<<"base at contact"<<std::endl;
        for(int i = 0; i<steps_indices.size(); i++){
          if(steps_indices[i] == contact.getPairObjectIndex()){ //c'e' stato un contatto tra la base e l'i-th step ///se nessun oggetto e' a contatto contact.getPairObjectIndex() ritorna 0
            return true;
          }
        }
      }
    }
      return false;
  }

  void curriculumUpdate() {//UYSO QUESTA FUNZIONE PERCHè viene chiamata solo una volta a fine episodio e aggiorna tutto

    generate_command_velocity();

    //Update the dimension of the steps
    for (int i=0; i<num_steps; i++){
      boxes[i] = world_->addBox(width_step -i, width_step-i, height_step, 10); //se trova gia' un oggetto nel punto in cui generarlo, lo mette sotto o sopra. 
      boxes[i]->setName("step" + std::to_string(i)+"-th");
      boxes[i]->setPosition(25,0,0.1*i);
      steps_indices[i] = boxes[i]->getIndexInWorld();  ///update the indices of the steps.
    }

    //rise slowly from 0,08 ->1
    curriculumFactor_ = std::pow(curriculumFactor_, curriculumDecayFactor_);

    if(number_of_falls_ >= 100)
      height_step = 0.75*std::pow(10, 1/curriculumDecayFactor_*log10(height_step));
    if(height_step < 0.25)
      height_step = std::pow(height_step, curriculumDecayFactor_);
    else 
      height_step = 0.25;

    RSINFO_IF(visualizable_, "Curriculum factor: "<< curriculumFactor_)
    RSINFO_IF(visualizable_, "height_step: "<<height_step)
    number_of_falls_ = 0; //reset the counter

   }

 private:
  int numberOfContact_ = 0;
  int num_seq;
  int num_legs_joint = 12;
  int gcDim_, gvDim_, nJoints_;
  int c_F_ = 1; //all'inizio, i piedi sono in contatto con il terreno. 
  int num_steps;
  double width_step, height_step;
  double curriculumFactor_, curriculumDecayFactor_;
  bool visualizable_ = false;
  bool twoTimeStep = false;
  bool nextTimeStep = false;
  raisim::ArticulatedSystem* anymal_;
  Eigen::VectorXd gc_init_, gv_init_, gc_, gv_, pTarget_, pTarget12_, vTarget_;
  Eigen::VectorXd ga_init_, ga_;
  double terminalRewardCoeff_ = -10.;
  //penalty reward
  double hipTerm_, thighTerm_, gaitTerm_Pos_, gaitTerm_Vel_;
  double timeDerivative = 0.001;

  Eigen::VectorXd actionMean_, actionStd_, obDouble_;

  Eigen::Vector3d bodyLinearVel_, bodyAngularVel_;
  std::set<size_t> footIndices_, footLinkFrame_;

  size_t baseIndex_;
  int number_of_falls_ = 0;
  std::vector<raisim::Box*> boxes;
  std::vector<size_t> steps_indices;
  bool update_height_step = false;
  /// these variables are not in use. They are placed to show you how to create a random number sampler.
  std::normal_distribution<double> normDist_;
  thread_local static std::mt19937 gen_;

  bool action_in_observation_space_ = false;
  bool joint_history_in_observation_space_ = false;
  Eigen::VectorXd previous_gv_;
  Eigen::VectorXd current_action_, previous_action_;

  Eigen::VectorXd joint_history_, joint_container_;
  Eigen::VectorXd joint_history_pos_, joint_history_vel_;
  Eigen::VectorXd q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11, q12;

  Eigen::VectorXd standing_configuration_;
  Eigen::VectorXd actual_joint_position_, actual_joint_velocities_;

  //footSlipage variables
  double tan_vel_norm = 0.0;
  double slip_term_ = 0.0;
  //std::vector<size_t> tan_vel_norm_vec;
  raisim::Vec<3> temp_footVelocity;

  //ClosedContact() (used even by footslipage)
  std::vector<int> cF_ = {1,1,1,1};


  Eigen::Vector3d command_;
};


thread_local std::mt19937 raisim::ENVIRONMENT::gen_;
}


