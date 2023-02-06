from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from .storage import RolloutStorage


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 num_envs,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=0.5,
                 entropy_coef=0.0,
                 learning_rate=5e-4,
                 max_grad_norm=0.5,
                 learning_rate_schedule='adaptive',
                 desired_kl=0.01,
                 use_clipped_value_loss=True,
                 log_dir='run',  #NON e' il percorso a run, ma solo la cartella. Se lo passi come argomento allora dai un percorso e puoi fare tutta la magagna sotto
                 device='cpu',
                 want_to_save_in_data_dir = False,
                 shuffle_batch=True):

        # PPO components
        self.actor = actor
        self.critic = critic
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor.obs_shape, critic.obs_shape, actor.action_shape, device)

        if shuffle_batch:
            self.batch_sampler = self.storage.mini_batch_generator_shuffle
        else:
            self.batch_sampler = self.storage.mini_batch_generator_inorder

        self.optimizer = optim.Adam([*self.actor.parameters(), *self.critic.parameters()], lr=learning_rate)
        self.device = device

        # env parameters
        self.num_transitions_per_env = num_transitions_per_env
        self.num_envs = num_envs

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        # Log
        directory = "tensorBoard_file"
        path_to_directory = log_dir
        """log_dir, per come e' stato passato come argomento nel runner.py, e' il percorso alla cartella all'interno di "data" nella cartella di raisimGymTorch"""
        self.log_dir = os.path.join(log_dir, directory) #join connect this 2 paths
        """Siccome se non passo log_dir, allora questo e' uguale a run, e NON AL PERCORSO A RUN, la cartella non sa dove crearla e quindi non glie la faccio
            creare. SE invece gli passo log_dir come saver.data_dir, allora sto passando un intero percorso, quindi sa dove crearmi la cartella"""
        if(want_to_save_in_data_dir == True): 
            os.mkdir(self.log_dir)
        """os.mkdir prende un percorso a una directory, e ti crea la directory in quel percorso se non esiste. Se gli dai come percorso "hom/claudio/directory"
        lui vede che in /home/claudio non esiste una cartella directory e quindi te la crea. Se invece quella cartella gia' esiste, ritorna errore"""
        #self.log_dir = os.path.join(log_dir, datetime.now().strftime('%b%d_%H-%M-%S')) #join connect this 2 paths

        #Self.log_dir a questo punto contiene il percorso allaa cartella "tensorBoard_file" in cui voglio salvare i file di tensorBoard
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10) #It takes as argument he save directory location. 
        #self.writer = SummaryWriter() 
        """Se a self.writer non specifichi una log_dir, te ne crea una lui dove hai lanciato il programma chiamata runs, e ti salva li i file per tensorBoard
        Se invece la specifichi, ti salva il file di tensorboard in quella cartella"""
        self.tot_timesteps = 0
        self.tot_time = 0

        # ADAM
        self.learning_rate = learning_rate
        self.desired_kl = desired_kl
        self.schedule = learning_rate_schedule

        # temps
        self.actions = None
        self.actions_log_prob = None
        self.actor_obs = None

    def act(self, actor_obs):
        self.actor_obs = actor_obs
        with torch.no_grad():
            self.actions, self.actions_log_prob = self.actor.sample(torch.from_numpy(actor_obs).to(self.device))
        return self.actions

    def step(self, value_obs, rews, dones):
        self.storage.add_transitions(self.actor_obs, value_obs, self.actions, self.actor.action_mean, self.actor.distribution.std_np, rews, dones,
                                     self.actions_log_prob)

    def update(self, actor_obs, value_obs, log_this_iteration, update):
        last_values = self.critic.predict(torch.from_numpy(value_obs).to(self.device))

        # Learning step
        self.storage.compute_returns(last_values.to(self.device), self.critic, self.gamma, self.lam)
        mean_value_loss, mean_surrogate_loss, infos = self._train_step(log_this_iteration)
        self.storage.clear()

        if log_this_iteration:
            self.log({**locals(), **infos, 'it': update})

    def log(self, variables):
        self.tot_timesteps += self.num_transitions_per_env * self.num_envs
        mean_std = self.actor.distribution.std.mean()
        self.writer.add_scalar('PPO/value_function', variables['mean_value_loss'], variables['it'])
        self.writer.add_scalar('PPO/surrogate', variables['mean_surrogate_loss'], variables['it'])
        self.writer.add_scalar('PPO/mean_noise_std', mean_std.item(), variables['it'])
        self.writer.add_scalar('PPO/learning_rate', self.learning_rate, variables['it'])

    def _train_step(self, log_this_iteration):
        mean_value_loss = 0
        mean_surrogate_loss = 0
        for epoch in range(self.num_learning_epochs):
            for actor_obs_batch, critic_obs_batch, actions_batch, old_sigma_batch, old_mu_batch, current_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch \
                    in self.batch_sampler(self.num_mini_batches):

                actions_log_prob_batch, entropy_batch = self.actor.evaluate(actor_obs_batch, actions_batch)
                value_batch = self.critic.evaluate(critic_obs_batch)

                # Adjusting the learning rate using KL divergence
                mu_batch = self.actor.action_mean
                sigma_batch = self.actor.distribution.std

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.no_grad():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.2)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.2)

                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                   1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = current_values_batch + (value_batch - current_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_([*self.actor.parameters(), *self.critic.parameters()], self.max_grad_norm)
                self.optimizer.step()

                if log_this_iteration:
                    mean_value_loss += value_loss.item()
                    mean_surrogate_loss += surrogate_loss.item()

        if log_this_iteration:
            num_updates = self.num_learning_epochs * self.num_mini_batches
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss, locals()
