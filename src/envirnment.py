import glob
import os
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.utility.utils import utility, return_next_item

class DeltaIotEnv(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"], "render_fps": 30}

    def __init__(self, n_actions, n_obs_space, data_dir, reward_type,
                 energy_coef=0.8, packet_coef=0.2, latency_coef=0.0,
                 energy_thresh=13.2, packet_thresh=15, latency_thresh=10, timesteps=216):
        super().__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(n_actions)
        # Example for using image as input (channel-first; channel-last also works):
        self.observation_space = spaces.Box(low=np.full(shape=n_obs_space, fill_value=-15, dtype=np.float32),
                                            high=np.full(
                                                shape=n_obs_space, fill_value=100, dtype=np.float32),
                                            dtype=float)
        self.data = data_dir
        self.info = {}
        self.data = return_next_item(self.data, normalize=False)
        self.reward = 0
        self.reward_type = reward_type()
        self.terminated = False
        self.truncated = False
        self.energy_coef = energy_coef
        self.packet_coef = packet_coef
        self.latency_coef = latency_coef
        self.energy_thresh = energy_thresh
        self.packet_thresh = packet_thresh
        self.latency_thresh = latency_thresh
        self.init_time_steps = timesteps

    def step(self, action):
        self.obs = self.df.iloc[action][[
            'energyconsumption', 'packetloss', 'latency']].to_numpy(dtype=float).flatten()

        energy_consumption = self.obs[0].flatten()
        packet_loss = self.obs[1].flatten()
        latency = self.obs[2].flatten()

        ut = utility(self.energy_coef, self.packet_coef, self.latency_coef,
                     energy_consumption, packet_loss, latency)

        self.reward = self.reward_type.get_reward(ut=ut, energy_consumption=energy_consumption,
                                                  packet_loss=packet_loss, latency=latency,
                                                  energy_thresh=self.energy_thresh, packet_thresh=self.packet_thresh,
                                                  latency_thresh=self.latency_thresh)
        self.time_steps -= 1
        if self.time_steps == 0:
            self.terminated = True
            self.truncated = True

        return self.obs, self.reward, self.terminated, self.truncated, self.info

    def reset(self, seed=None, options=None):
        self.time_steps = self.init_time_steps
        self.terminated = False
        self.truncated = False
        self.df = next(self.data)
        rand_num = np.random.randint(self.df.count().iloc[0])
        self.obs = self.df.iloc[rand_num][[
            'energyconsumption', 'packetloss', 'latency']].to_numpy(dtype=float).flatten()
        return self.obs, self.info

    def render(self):
        ...

    def close(self):
        ...
