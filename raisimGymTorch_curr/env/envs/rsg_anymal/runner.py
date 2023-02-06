from ruamel.yaml import YAML, dump, RoundTripDumper
import os
from raisimGymTorch.env.RaisimGymVecEnv import RaisimGymVecEnv as VecEnv
from raisimGymTorch.helper.raisim_gym_helper import ConfigurationSaver, load_param, tensorboard_launcher
from raisimGymTorch.env.bin.rsg_anymal import NormalSampler
from raisimGymTorch.env.bin.rsg_anymal import RaisimGymEnv
import math
import time
import torch


import raisimGymTorch.algo.ppo.module as ppo_module
import raisimGymTorch.algo.ppo.ppo as PPO
import torch.nn as nn
import numpy as np
import datetime
import argparse

# task specification
task_name = "anymal_locomotion"
anymal_path = "/home/claudio/raisim_ws/raisimlib/rsc/anymal/urdf/anymal.urdf"
# configuration
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', help='set mode either train or test', type=str, default='train') #-m is what you have to write in command line to specify this argument. 
parser.add_argument('-w', '--weight', help='pre-trained weight path', type=str, default='')
#parser.add_argument('-b', '--tensorboard', help='launch tensorboard or not', type=bool, default='False')
args = parser.parse_args() #This is needed to assign the value from command line to local variables
mode = args.mode #Thanks to args, we can access to the value of the variable spacified by command line and assign it to a local variable.
weight_path = args.weight
#Use_tensorboard = args.tensorboard

# check if gpu is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# directories
task_path = os.path.dirname(os.path.realpath(__file__))
home_path = task_path + "/../../../../.."  #Path to raisim folder (where there is GymTorch, Unity etc)
path_where_launch_script = home_path + "/raisimGymTorch"

# config
cfg = YAML().load(open(task_path + "/cfg.yaml", 'r'))

# create environment from the configuration file
env = VecEnv(RaisimGymEnv(home_path + "/rsc", dump(cfg['environment'], Dumper=RoundTripDumper))) #here you are colling the constructor of Vectorized Environment because you are constructing the object env
env.seed(cfg['seed'])

# shortcuts
ob_dim = env.num_obs
act_dim = env.num_acts
num_threads = cfg['environment']['num_threads']

# Training
n_steps = math.floor(cfg['environment']['max_time'] / cfg['environment']['control_dt'])
total_steps = n_steps * env.num_envs

avg_rewards = []

actor = ppo_module.Actor(ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim),
                         ppo_module.MultivariateGaussianDiagonalCovariance(act_dim,
                                                                           env.num_envs,
                                                                           1.0,
                                                                           NormalSampler(act_dim),
                                                                           cfg['seed']),
                         device)
critic = ppo_module.Critic(ppo_module.MLP(cfg['architecture']['value_net'], nn.LeakyReLU, ob_dim, 1),
                           device)

saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=[task_path + "/cfg.yaml", task_path + "/Environment.hpp"], create_dir = True)

"""save_items = None
saver = ConfigurationSaver(log_dir=home_path + "/raisimGymTorch/data/"+task_name,
                           save_items=save_items, create_dir=False)
if(save_items is not None):"""
tensorboard_launcher(saver.data_dir+"/..")  # press refresh (F5) after the first ppo update



ppo = PPO.PPO(actor=actor,
              critic=critic,
              num_envs=cfg['environment']['num_envs'],
              num_transitions_per_env=n_steps,
              num_learning_epochs=4,
              gamma=0.996,
              lam=0.95,
              num_mini_batches=4,
              device=device,
              log_dir=saver.data_dir, #log_dir sara' il percorso alla cartella all'interno della folder "data/anymal_locomotion/January_bla_bla"
              shuffle_batch=False,
              want_to_save_in_data_dir = True
              )

if mode == 'retrain':
    load_param(weight_path, env, actor, critic, ppo.optimizer, saver.data_dir)

EvalThePolicy = False
for update in range(10000):
    start = time.time()
    env.reset()
    reward_ll_sum = 0
    done_sum = 0
    average_dones = 0.


    #For visualization. Skip the if the see the training part
    if (update % cfg['environment']['eval_every_n'] == 0 and EvalThePolicy):
        print("Visualizing and evaluating the current policy")
        torch.save({
            'actor_architecture_state_dict': actor.architecture.state_dict(), #gli sto passando un dizionario a torch.save {'chiave': valore, }
            'actor_distribution_state_dict': actor.distribution.state_dict(), #state dict restituisce un dizionario, in cui la chiave e' il nome del layer e il valore e' il tensore dei parametri
            'critic_architecture_state_dict': critic.architecture.state_dict(),
            'optimizer_state_dict': ppo.optimizer.state_dict(),
        }, saver.data_dir+"/full_"+str(update)+'.pt') #il secondo parametro e' il nome del file che sto salvando e che potro' ricaricare con load. Tutto quel bordello, e' semplicemente un percorso piu' il file.pt in cui si sono i parametri
        #Qui gli stiamo solo dicendo che deve salvarsi quei dizionari (dove la chiave e' il nome del layer e il valore sono i parametri del layer) da qualche parte come file .pt
        
        # we create another graph just to demonstrate the save/load method. Useremo questa poliicy per far vedere in quel momento come si comporta la policy addestrata
        loaded_graph = ppo_module.MLP(cfg['architecture']['policy_net'], nn.LeakyReLU, ob_dim, act_dim)
        loaded_graph.load_state_dict(torch.load(saver.data_dir+"/full_"+str(update)+'.pt')['actor_architecture_state_dict'])

        env.turn_on_visualization()  #se dopo lo iberno, ora devo risvegliarlo
        env.start_video_recording(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + "policy_"+str(update)+'.mp4')

        for step in range(n_steps*2):
            with torch.no_grad():
                frame_start = time.time() #riporta il tempo attuale come un float espresso in secondi. per avere l'ora attuale devi usare  print (time.asctime( time.localtime(time.time()) )). 
                obs = env.observe(False)
                action_ll = loaded_graph.architecture(torch.from_numpy(obs).cpu())  #Usa l'actor che in quello step ti sei salvato, cioe' quello che e' stato addestrato  fino a quel momento
                reward_ll, dones = env.step(action_ll.cpu().detach().numpy())
                frame_end = time.time()
                wait_time = cfg['environment']['control_dt'] - (frame_end-frame_start)
                if wait_time > 0.:
                    time.sleep(wait_time)  #sleep untile the next control_step
                    #time.sleep(0.01) #sleep takes the seconds

        env.stop_video_recording()
        env.turn_off_visualization()  #iberna il server, non voglio vedere piu' il robot su Unity muoversi

        env.reset()
        env.save_scaling(saver.data_dir, str(update))

    EvalThePolicy = True
    # actual training
    for step in range(n_steps):
        obs = env.observe() 
        action = ppo.act(obs)
        #frame_start1 = time.time()
        reward, dones = env.step(action)
        #frame_end1 = time.time()
        #print(frame_end1 - frame_start1) Non ha senso questa valutazione, il sistema non e' real time, il tempo trascorso non e' uguale al tempo che ci mette il simulatore
        ppo.step(value_obs=obs, rews=reward, dones=dones)
        done_sum = done_sum + np.sum(dones)
        reward_ll_sum = reward_ll_sum + np.sum(reward)

    # take st step to get value obs
    obs = env.observe()
    ppo.update(actor_obs=obs, value_obs=obs, log_this_iteration=update % 10 == 0, update=update)
    average_ll_performance = reward_ll_sum / total_steps
    average_dones = done_sum / total_steps
    avg_rewards.append(average_ll_performance)

    actor.update()
    actor.distribution.enforce_minimum_std((torch.ones(12)*0.2).to(device))

    # curriculum update. Implement it in Environment.hpp
    env.curriculum_callback()

    end = time.time()

    print('----------------------------------------------------')
    print('{:>6}th iteration'.format(update))
    print('{:<40} {:>6}'.format("average ll reward: ", '{:0.10f}'.format(average_ll_performance)))
    print('{:<40} {:>6}'.format("dones: ", '{:0.6f}'.format(average_dones)))
    print('{:<40} {:>6}'.format("time elapsed in this iteration: ", '{:6.4f}'.format(end - start)))
    print('{:<40} {:>6}'.format("fps: ", '{:6.0f}'.format(total_steps / (end - start))))
    print('{:<40} {:>6}'.format("real time factor: ", '{:6.0f}'.format(total_steps / (end - start)
                                                                       * cfg['environment']['control_dt'])))
    print('----------------------------------------------------\n')