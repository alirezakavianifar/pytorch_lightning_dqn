import copy
import gymnasium as gym
import torch

import numpy as np
import torch.nn.functional as F

from collections import deque, namedtuple

from torch import Tensor
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset
from torch.optim import AdamW

from pytorch_lightning import LightningDataModule, Trainer

from gymnasium.wrappers import record_episode_statistics

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

