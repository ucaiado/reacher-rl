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
from collections import namedtuple, deque

import torch
import torch.nn.functional as F
import torch.optim as optim

from .models import Actor, Critic

import pdb

'''
Begin help functions and variables
'''


BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 10         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor
LR_CRITIC = 3e-4        # learning rate of the critic
WEIGHT_DECAY = 0.0001   # L2 weight decay
UPDATE_EVERY = 20       # steps to update


devc = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
End help functions and variables
'''


class MetaAgent(object):
    '''
    '''

    def __init__(self, state_size, action_size, nb_agents, rand_seed):
        '''Initialize an MetaAgent object.

        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param nb_agents: int. number of agents to use
        :param seed: int. random seed
        '''
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE,
                                   BATCH_SIZE, rand_seed)
        self.nb_agents = nb_agents
        self.action_size = action_size
        self.l_agents = [DDPGAgent(state_size, action_size, rand_seed,
                                   self.memory)
                         for i in range(nb_agents)]

    def step(self, states, actions, rewards, next_states, dones):
        iter_obj = zip(self.l_agents, states, actions, rewards, next_states,
                       dones)
        for agent, state, action, reward, next_state, done in iter_obj:
            agent.step(state, action, reward, next_state, done)

    def act(self, states, add_noise=True):
        na_rtn = np.zeros([self.nb_agents, self.action_size])
        for idx, agent in enumerate(self.l_agents):
            na_rtn[idx, :] = agent.act(states[idx], add_noise)
        return na_rtn

    def reset(self):
        for agent in self.l_agents:
            agent.reset()


class DDPGAgent(object):
    '''
    Implementation of a DQN agent that interacts with and learns from the
    environment
    '''

    def __init__(self, state_size, action_size, rand_seed, memory):
        '''Initialize an DDPGAgent object.

        :param state_size: int. dimension of each state
        :param action_size: int. dimension of each action
        :param seed: int. random seed
        :param memory: ReplayBuffer object.
        '''
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(rand_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, rand_seed).to(devc)
        self.actor_target = Actor(state_size, action_size, rand_seed).to(devc)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(),
                                          lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic_local = Critic(state_size, action_size, rand_seed).to(devc)
        self.critic_target = Critic(state_size, action_size, rand_seed).to(devc)
        # NOTE: the decay corresponds to L2 regularization
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=LR_CRITIC,
                                           weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, rand_seed)

        # Replay memory
        self.memory = memory

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
                # source: Sample a random minibatch of N transitions from R
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, add_noise=True):
        '''Returns actions for given state as per current policy.

        :param state: array_like. current state
        :param add_noise: Boolean. If should add noise to the action
        '''
        state = torch.from_numpy(state).float().unsqueeze(0).to(devc)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        # source: Select action at = μ(st|θμ) + Nt according to the current
        # policy and exploration noise
        if add_noise:
            action += self.noise.sample()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

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
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
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


class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x)
        dx += self.sigma * np.array([random.random() for i in range(len(x))])
        self.state = x + dx
        return self.state


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
