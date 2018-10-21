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
import torch.nn as nn
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
TAU = None
BATCH_SIZE = None
GRADIENT_CLIP = None


class Batcher:

    def __init__(self, batch_size, data):
        self.batch_size = batch_size
        self.data = data
        self.num_entries = len(data[0])
        self.reset()

    def reset(self):
        self.batch_start = 0
        self.batch_end = self.batch_start + self.batch_size

    def end(self):
        return self.batch_start >= self.num_entries

    def next_batch(self):
        batch = []
        for d in self.data:
            batch.append(d[self.batch_start: self.batch_end])
        self.batch_start = self.batch_end
        self.batch_end = min(self.batch_start + self.batch_size, self.num_entries)
        return batch

    def shuffle(self):
        indices = np.arange(self.num_entries)
        np.random.shuffle(indices)
        self.data = [d[indices] for d in self.data]


def set_global_parms(d_table):
    '''
    convert statsmodel tabel to the agent parameters

    :param d_table: Dictionary. Parameters of the agent
    '''
    global LR, SGD_EPOCH, BETA, EPSILON, DISCOUNT, DEVC, PARAMS, TAU
    global BATCH_SIZE, GRADIENT_CLIP
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
    TAU = d_params['TAU']
    BATCH_SIZE = d_params['BATCH_SIZE']
    GRADIENT_CLIP = d_params['GRADIENT_CLIP']
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
                # print('\n\nnew epoch')
                # experiences = self.memory.sample()
                self.learn(trajectory, eps, beta)

    def act(self, states):
        '''Returns actions for given states as per current policy.

        :param states: array_like. current states
        '''
        states = torch.from_numpy(states).float().to(DEVC)
        self.policy.eval()
        with torch.no_grad():
            actions, log_probs, entropy_loss, values = self.policy(states)
            actions = actions.cpu().data.numpy()
            log_probs = log_probs.cpu().data.numpy()
            values = values.cpu().data.numpy()
        self.policy.train()  # ?
        return actions, log_probs, values

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
        dones = torch.Tensor(trajectories['done'])
        nb = self.nb_agents

        # calculate the advatages
        # processed_rollout = [None] * (len(dones))
        # advantages = torch.Tensor(np.zeros((self.nb_agents, 1)))
        # i_max = len(states)
        # for i in reversed(range(i_max)):
        #     terminals_ = torch.Tensor(dones[i]).unsqueeze(1)
        #     rwrds_ = torch.Tensor(rewards[i]).unsqueeze(1)
        #     values_ = torch.Tensor(old_values[i])
        #     next_value_ = old_values[min(i_max-1, i + 1)]

        #     td_error = rwrds_ + DISCOUNT * terminals_ * next_value_.detach()
        #     td_error -= values_.detach()
        #     advantages = advantages * TAU * DISCOUNT * terminals_ + td_error
        #     processed_rollout[i] = advantages

        # advantages = torch.stack(processed_rollout).squeeze(2)
        # advantages = (advantages - advantages.mean()) / advantages.std()

        # learn in batches
        # batcher = Batcher(states.size(0) // 16, [np.arange(states.size(0))])
        batcher = Batcher(BATCH_SIZE, [np.arange(states.size(0))])
        batcher.shuffle()
        while not batcher.end():
            batch_indices = batcher.next_batch()[0]
            batch_indices = torch.Tensor(batch_indices).long()

            this_loss = clipped_surrogate(self.policy,
                                          old_probs[batch_indices],
                                          states[batch_indices],
                                          actions[batch_indices],
                                          rewards[batch_indices],
                                          old_values[batch_indices],
                                          # advantages[batch_indices],
                                          0,
                                          nb,
                                          eps,
                                          beta)
            # Minimize the loss
            self.optimizer.zero_grad()
            this_loss.backward()
            # nn.utils.clip_grad_norm_(self.policy.parameters(), GRADIENT_CLIP)
            self.optimizer.step()


def clipped_surrogate(policy, old_probs, states, actions, rewards, old_values,
                      advantages, nb_agents, epsilon=0.1, beta=0.01):
        # discount rewards and convert them to future rewards
        discount = DISCOUNT**np.arange(len(rewards))
        rewards = np.asarray(rewards)*discount[:, np.newaxis]
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

        _, new_probs, entropy_loss, values = policy(states, actions)

        # ratio for clipping. All probabilities used are log probabilities
        # with torch.no_grad():
        ratio = (new_probs - old_probs).exp()
        # print(torch.max(ratio), torch.min(ratio))
        # pdb.set_trace()

        # clipped function
        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_surrog = torch.min(ratio*rewards[:, :, np.newaxis],
                                   clip*rewards[:, :, np.newaxis])
        # clipped_surrog = torch.min(ratio*advantages[:, :, np.newaxis],
        #                            clip*advantages[:, :, np.newaxis])

        # include a regularization term
        # this steers new_policy towards 0.5
        # this returns an average of all the entries of the tensor
        # effective computing L_sur^clip / T
        # averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards
        entropy_loss = entropy_loss[:, :, np.newaxis]
        policy_loss = torch.mean(clipped_surrog + beta*entropy_loss)
        return -policy_loss
