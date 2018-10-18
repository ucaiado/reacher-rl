#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement a model-free approach called Deep DPG (DDPG)


@author: udacity, ucaiado

Created on 10/07/2018
"""

import numpy as np
import random
import copy
import os
import yaml
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

import pdb

try:
    from agent_utils import param_table, Actor, Critic, Policy
except:
    from .agent_utils import param_table, Actor, Critic, Policy

'''
Begin help functions and variables
'''
BUFFER_SIZE = None
BATCH_SIZE = None
GAMMA = None
TAU = None
LR_ACTOR = None
LR_CRITIC = None
WEIGHT_DECAY = None
UPDATE_EVERY = None
DEVC = None
PARAMS = None


# Discount rate - 0.99
# Tau - 0.95
# Rollout length - 2048
# Optimization epochs - 10
# Gradient clip - 0.2
# Learning rate - 3e-4


def set_global_parms(d_table):
    '''
    convert statsmodel tabel to the agent parameters

    :param d_table: Dictionary. Parameters of the agent
    '''
    global BUFFER_SIZE, BATCH_SIZE, GAMMA, TAU, LR_ACTOR, LR_CRITIC
    global WEIGHT_DECAY, UPDATE_EVERY, DEVC, PARAMS
    l_table = [(a, [b]) for a, b in d_table.items()]
    d_params = dict([[x[0], x[1][0]] for x in l_table])
    table = param_table.generate_table(l_table[:int(len(l_table)/2)],
                                       l_table[int(len(l_table)/2):],
                                       'PPO PARAMETERS')
    GAMMA = d_params['GAMMA']                 # discount factore
    DEVC = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    PARAMS = table

PATH = os.path.dirname(os.path.realpath(__file__))
PATH = PATH.replace('ppo', 'config.yaml')
set_global_parms(yaml.load(open(PATH, 'r'))['PPO'])

'''
End help functions and variables
'''


class Agent(object):
    '''
    Implementation of a DQN agent that interacts with and learns from the
    environment
    '''

    def __init__(self, state_size, action_size, nb_agents, rand_seed):
        '''Initialize an MetaAgent object.

        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param nb_agents: int. number of agents to use
        :param seed: int. random seed
        '''

        self.nb_agents = nb_agents
        self.action_size = action_size
        self.__name__ = 'PPO'

        # Policy Network
        action_net = Actor(state_size, action_size, rand_seed).to(DEVC)
        critic_net = Critic(state_size, action_size, rand_seed).to(DEVC)
        self.policy = Policy(action_size, action_net, critic_net)
        self.optimizer = optim.Adam(self.actor_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE,
                                   BATCH_SIZE, rand_seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, states, actions, rewards, next_states, dones):

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

    def act(self, states):
        '''Returns actions for given states as per current policy.

        :param states: array_like. current states
        '''
        states = torch.from_numpy(states).float().to(DEVC)
        self.policy.eval()
        with torch.no_grad():
            actions, log_probs, _, values = self.policy(states)
            actions = actions.cpu().data.numpy()
            log_probs = log_probs.cpu().data.numpy()
            values = values.cpu().data.numpy()
        # pdb.set_trace()
        if add_noise:
            actions += self.noise.sample()
        return actions, log_probs, values

    def learn(self, experiences, gamma):
        '''
        Update policy and value params using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        :param experiences: Tuple[torch.Tensor]. tuple of (s, a, r, s', done)
        :param gamma: float. discount factor
        '''
        states, actions, rewards, next_states, dones = experiences
        # rewards_ = torch.clamp(rewards, min=-1., max=1.)
        rewards_ = rewards

        # --------------------------- update critic ---------------------------
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards_ + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss: L = 1/N SUM{(yi − Q(si, ai|θQ))^2}
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # suggested by Attempt 3, from Udacity
        # torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # --------------------------- update actor ---------------------------
        # Compute actor loss: ∇θμ J ≈1/N  ∇aQ(s, a|θQ)|s=si,a=μ(si)∇θμ μ(s|θμ)
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ---------------------- update target networks ----------------------
        # Update the critic target networks: θQ′ ←τθQ +(1−τ)θQ′
        # Update the actor target networks: θμ′ ←τθμ +(1−τ)θμ′
        self.soft_update(self.critic_local, self.critic_target, TAU)
        self.soft_update(self.actor_local, self.actor_target, TAU)






def clipped_surrogate(policy, old_probs, states, actions, rewards,
                      discount=0.995,
                      epsilon=0.1, beta=0.01):

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]

    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]

    # convert everything into pytorch tensors and move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=device)

    # convert states to policy (or probability)
    new_probs = states_to_prob(policy, states)
    new_probs = torch.where(actions == RIGHT, new_probs, 1.0-new_probs)

    # ratio for clipping
    ratio = new_probs/old_probs

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

    # include a regularization term
    # this steers new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+ \
        (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))


    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta*entropy)
