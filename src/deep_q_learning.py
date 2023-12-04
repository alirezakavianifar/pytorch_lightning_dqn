import copy
import torch

import numpy as np
import torch.nn.functional as F

from collections import deque, namedtuple

from torch import Tensor
from torch.utils.data import DataLoader
from torch.optim import AdamW

from pytorch_lightning import LightningModule, LightningDataModule

from gymnasium.wrappers import record_episode_statistics

from policy import epsilon_greedy
from network import DQNNet
from replay_buffer import ReplayBuffer, RLDataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


class DeepQLearning(LightningModule):
    def __init__(self, env, policy=epsilon_greedy, capacity=100_000,
                 batch_size=256, lr=1e-3, hidden_size=128, gamma=0.99,
                 loss_fn=F.smooth_l1_loss, optim=AdamW, eps_start=1.0, eps_end=0.15,
                 eps_last_episode=100, samples_per_epoch=256, sync_rate=10):

        self.env = env

        obs_size = env.observation_space.shape[0]
        n_actions = self.env.action_space.n

        self.q_net = DQNNet(hidden_size=hidden_size, obs_size=obs_size, n_actions=n_actions)

        self.target_q_net = copy.deepcopy(self.q_net)

        self.policy = policy
        self.buffer = ReplayBuffer(capacity=capacity)

        self.save_hyperparameters()

        while len(self.buffer) < self.hparams.samples_per_epoch:
            print(f'{len(self.buffer)} samples in experience buffer. Filling ...')
            self.play_episode(epsilon=self.hparams.eps_start)

    def play_episode(self, policy=None, epsilon=0.):
        state = self.env.reset()
        done = False

        while not done:
            if policy:
                action = policy(state, self.env, self.q_net, epsilon=epsilon)
            else:
                action = self.env.action_space.sample()

            next_state, reward, done, info = self.env.step(action)
            exp = (state, action, reward, done, next)
            self.buffer.append(exp)
            state = next_state

    def forward(self, x):
        return self.q_net(x)
    
    def configure_optimizers(self):
        q_net_optimizer = self.hparams.optim(
            self.q_net.parameters(), 
            lr=self.hparams.lr
        )
        return [q_net_optimizer]
    
    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.samples_per_epoch)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size
        )

        return dataloader
    
    def training_step(self, batch, batch_idx):
        states, actions, rewards, dones, next_states = batch
        actions = actions.unsqueeze(1)
        rewards = rewards.unsqueeze(1)
        dones = dones.unsqueeze(1)

        state_action_values = self.q_net(states).gather(1, actions)

        next_action_values, _ = self.target_q_net(next_states).max(dim=1, keepdim=True)
        next_action_values[dones] = 0.0

        expected_state_action_values = rewards + self.hparams.gamma * next_action_values

        loss = self.hparams.loss_fn(state_action_values, expected_state_action_values)
        self.log('episode/Q-Error', loss)
        return loss
    
    def training_epoch_end(self):

        epsilon = max(
            self.hparams.eps_end,
            self.hparams.eps_start - self.current_epoch / self.hparams.eps_last_episode
        )

        self.play_episode(policy=self.policy, epsilon=epsilon)
        self.log('episode/Return', self.env.return_queue[-1])

        if self.current_epoch % self.hparams.sync_rate == 0:
            self.target_q_net.load_state_dict(self.q_net.state_dict())












