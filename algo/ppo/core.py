import numpy as np
import scipy.signal
from gym.spaces import Box, Discrete

import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.categorical import Categorical
import torch.nn.functional as F

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

class cnn_net(nn.Module):
    def __init__(self, num_inputs, num_out, activation=nn.ReLU):
        super(cnn_net, self).__init__()
        self.conv1 = nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.fc_in = nn.Linear(32*6*6, 512)
        self.fc_out = nn.Linear(512, num_out)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LSTMCell):
                nn.init.constant_(module.bias_ih, 0)
                nn.init.constant_(module.bias_hh, 0)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc_in(x))
        out = self.fc_out(x)
        return out

def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])

def discount_cumsum(x, discount):
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


class MarioActor(nn.Module):
    
    def __init__(self, obs_dim, act_dim, activation):
        super().__init__()
        self.logits_net = cnn_net(obs_dim,act_dim,activation).to(device)

    def forward(self, obs, act=None):
        logits = self.logits_net(obs)
        pi = Categorical(logits=logits)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a

class MarioCritic(nn.Module):

    def __init__(self, obs_dim, activation):
        super().__init__()
        self.v_net = cnn_net(obs_dim,1,activation).to(device)

    def forward(self, obs):
        return torch.squeeze(self.v_net(obs), -1) # Critical to ensure v has right shape.

# class MarioActorCritic(nn.Module):

#     def __init__(self, observation_space, action_space, activation=nn.ReLU):
#         super().__init__()
#         obs_dim = observation_space.shape[0]
#         self.pi = MarioActor(obs_dim, action_space.n, activation)
#         self.v  = MarioCritic(obs_dim, activation)

#     def step(self, obs):
#         with torch.no_grad():
#             pi = self.pi._distribution(obs)
#             a = pi.sample()
#             logp_a = self.pi._log_prob_from_distribution(pi, a)
#             v = self.v(obs)
#         return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

#     def act(self, obs):
#         return self.step(obs)[0]