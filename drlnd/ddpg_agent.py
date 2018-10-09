#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement bla


@author: ucaiado

Created on 10/07/2018
"""

import numpy as np
import random
import copy
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from models import Actor, Critic

'''
Begin help functions and variables'
'''


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
End help functions and variables'
'''


class Agent(object):
    '''
    Implementation of a DQN agent that interacts with and learns from the
    environment
    '''

    def __init__(self, state_size, action_size, random_seed):
        '''Initialize an DQNAgent object.

        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param seed: int. random seed
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnet = QNetwork(state_size, action_size, seed).to(devc)
        self.qnet_target = QNetwork(state_size, action_size, seed).to(devc)
        # TODO: test RMSprop here
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset
            # and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        '''Returns actions for given state as per current policy.

        :param state: array_like. current state
        :param eps: float. epsilon, for epsilon-greedy action selection
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(devc)
        self.qnet.eval()
        with torch.no_grad():
            action_values = self.qnet(state)
        self.qnet.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def reset(self):
        pass

    def learn(self, experiences, gamma):
        '''Update value parameters using given batch of experience tuples.

        :param experiences: Tuple[torch.Tensor]. tuple of (s, a, r, s', done)
        :param gamma: float. discount factor
        '''
        states, actions, rewards, next_states, dones = experiences
        rewards_ = torch.clamp(rewards, min=-1., max=1.)
        rewards_ = rewards

        # it is optimized to use in gpu
        # Get max predicted Q values (for next states) from target model
        # max(1) := np.max(array, axis=1)
        # unsqueeze : = array.reshape(-1, 1)
        # max_a' \hat{Q}(\phi(s_{t+1}, a'; θ^{-}))
        max_Qhat = self.qnet_target(next_states).detach().max(1)[0].unsqueeze(1)
        # y_i = r + γ * maxQhat
        # y_i = r, if done
        Q_target = rewards_ + (gamma * max_Qhat * (1 - dones))

        # Q(\phi(s_t), a_j; \theta)
        Q_expected = self.qnet(states).gather(1, actions)

        # perform gradient descent step on on (y_i - Q)**2
        loss = F.mse_loss(Q_expected, Q_target)
        self.optimizer.zero_grad()  # Clear the gradients
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnet, self.qnet_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        '''Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: PyTorch model. weights will be copied from
        :param target_model: PyTorch model. weights will be copied to
        :param tau: float. interpolation parameter
        '''
        iter_params = zip(target_model.parameters(), local_model.parameters())
        for target_param, local_param in iter_params:
            tensor_aux = tau*local_param.data + (1.0-tau)*target_param.data
            target_param.data.copy_(tensor_aux)


class ReplayBuffer(object):
    '''Fixed-size buffer to store experience tuples.'''

    def __init__(self, action_size, buffer_size, batch_size, seed):
        '''Initialize a ReplayBuffer object.

        :param action_size: int. dimension of each action
        :param buffer_size: int: maximum size of buffer
        :param batch_size (int): size of each training batch
        :param seed (int): random seed
        '''
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience",
                                     field_names=["state", "action", "reward",
                                                  "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        '''Add a new experience to memory.'''
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        '''Randomly sample a batch of experiences from memory.'''
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences
                                  if e is not None])).float().to(devc)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences
                                   if e is not None])).long().to(devc)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences
                                   if e is not None])).float().to(devc)
        next_states = torch.from_numpy(np.vstack([e.next_state
                                                  for e in experiences
                                                  if e is not None])).float().to(devc)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences
                                            if e is not None]).astype(np.uint8)).float().to(devc)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        '''Return the current size of internal memory.'''
        return len(self.memory)
