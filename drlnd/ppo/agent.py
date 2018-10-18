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
LR = None
SGD_EPOCH = None
DISCOUNT = None
EPSILON = None
BETA = None
UPDATE_EVERY = 1
DEVC = None
PARAMS = None


# Discount rate - 0.99
# Tau - 0.95
# Rollout length - 2048
# Optimization epochs - 10
# Gradient clip - 0.2
# Learning rate - 3e-4


def states_to_prob(policy, states):
    '''
    convert states to probability, passing through the polic

    :param policy: Torch NNetowrk.
    :param states: Torch Tensor.
    '''
    states = torch.stack(states)
    policy_input = states.view(-1, *states.shape[-3:])
    _, log_probs, _, _ = policy(policy_input)
    return log_probs.view(states.shape[:-3])


def set_global_parms(d_table):
    '''
    convert statsmodel tabel to the agent parameters

    :param d_table: Dictionary. Parameters of the agent
    '''
    global LR, SGD_EPOCH, BETA, EPSILON, DISCOUNT, DEVC, PARAMS
    l_table = [(a, [b]) for a, b in d_table.items()]
    d_params = dict([[x[0], x[1][0]] for x in l_table])
    table = param_table.generate_table(l_table[:int(len(l_table)/2)],
                                       l_table[int(len(l_table)/2):],
                                       'PPO PARAMETERS')
    LR = d_params['LR']
    SGD_EPOCH = d_params['SGD_EPOCH']
    BETA = d_params['BETA']
    EPSILON = d_params['EPSILON']
    DISCOUNT = d_params['DISCOUNT']
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
        actor_net = Actor(state_size, action_size, rand_seed).to(DEVC)
        critic_net = Critic(state_size, action_size, rand_seed).to(DEVC)
        self.policy = Policy(action_size, actor_net, critic_net)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=LR)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, trajectory, eps, beta):

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            for i in range(SGD_EPOCH):
                # experiences = self.memory.sample()
                self.learn(trajectory, eps, beta)

    def act(self, states):
        '''Returns actions for given states as per current policy.

        :param states: array_like. current states
        '''
        states = torch.from_numpy(states).float().to(DEVC)
        self.policy.eval()
        with torch.no_grad():
            actions, log_probs, other1, values = self.policy(states)
            actions = actions.cpu().data.numpy()
            log_probs = log_probs.cpu().data.numpy()
            values = values.cpu().data.numpy()
        # self.policy.train()  # ?
        return np.clip(actions, -1, 1), log_probs, values

    def learn(self, trajectories, eps, beta):
        '''
        Update policy and value params using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        :param trajectories: Trajectory object. tuples of (s, a, r, s', done)
        :param eps: float. epsilon
        :param beta: float. beta
        '''

        states = torch.Tensor(trajectories['state'])
        rewards = torch.Tensor(trajectories['reward'])
        old_probs = torch.Tensor(trajectories['prob'])
        actions = torch.Tensor(trajectories['action'])
        old_values = torch.Tensor(trajectories['value'])
        this_loss = clipped_surrogate(self.policy, old_probs, states, actions,
                                      rewards, old_values, eps, beta)
        # Minimize the loss
        self.optimizer.zero_grad()
        this_loss.backward()
        self.optimizer.step()


def clipped_surrogate(policy, old_probs, states, actions, rewards, old_values,
                      epsilon=0.1, beta=0.01):
        # discount rewards and convert them to future rewards
        discount = DISCOUNT**np.arange(len(rewards))
        print(DISCOUNT)
        print(DISCOUNT**np.arange(len(rewards)))
        pdb.set_trace()
        rewards = np.asarray(rewards)[:, np.newaxis]*discount[:, np.newaxis]
        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
        mean = np.mean(rewards_future, axis=1)
        std = np.std(rewards_future, axis=1) + 1.0e-10
        rwds_normalized = (rewards_future - mean[:, np.newaxis])
        rwds_normalized /= std[:, np.newaxis]

        # convert everything into pytorch tensors and move to gpu if available
        actions = torch.tensor(actions, dtype=torch.float, device=DEVC)
        old_probs = torch.tensor(old_probs, dtype=torch.float, device=DEVC)
        old_values = torch.tensor(old_values, dtype=torch.float, device=DEVC)
        rewards = torch.tensor(rwds_normalized, dtype=torch.float, device=DEVC)

        decided_act, new_probs, entropy_loss, values = policy(states, actions)

        # ratio for clipping
        ratio = (new_probs - old_probs).exp()

        # clipped function
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_surrog = torch.min(ratio*rewards, clip*rewards)
        # pdb.set_trace()

        # include a regularization term
        # this steers new_policy towards 0.5
        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        policy_loss = -torch.mean(clipped_surrog + beta*entropy_loss)
        value_loss = 0.5 * (old_values - values).pow(2).mean()
        # print(policy_loss)
        return policy_loss
