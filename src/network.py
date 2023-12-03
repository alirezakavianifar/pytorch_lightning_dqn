import torch.nn as nn

class DQNNet(nn.Module):
    def __init__(self, hidden_size, obs_size, n_actions) -> None:
        self.net = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(obs_size, n_actions),
        )

    def forward(self, x):
        return self.net(x.float())

