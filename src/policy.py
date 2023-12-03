import numpy as np
import torch

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
def epsilon_greedy(state, env, net, epsilon=0.0):

    if np.random() < epsilon:
        action = env.action_space.sample()
    else:
        state = torch.tensor([state]).to(device)
        q_values = net(state)
        _, action = torch.max(q_values, dim=1)
        action = int(action.item())
    return action
