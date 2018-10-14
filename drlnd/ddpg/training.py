#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement ...


@author: ucaiado

Created on 10/07/2018
"""
from agent import Agent, PARAMS
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
import platform
import time
import pickle
import pdb
import torch
from agent_utils import make


'''
Begin help functions and variables
'''

DATA_PREFIX = '../../data/2018-10-07-'
SOLVED = False


'''
End help functions and variables
'''

if __name__ == '__main__':
    env = make()

    # from drlnd.ddpg_agent import Agent
    episodes = 3
    rand_seed = 0

    scores = []
    scores_std = []
    scores_avg = []
    scores_window = deque(maxlen=100)  # last 100 scores

    agent = Agent(env.state_size, env.action_size, env.num_agents, rand_seed)

    print('\n')
    print(PARAMS)

    print('\nNN ARCHITECURES:')
    print(agent.actor_local)
    print(agent.critic_local)

    print('\nTRAINING:')
    for episode in range(episodes):
        states = env.reset()
        score = 0.
        for i in range(1000):
            actions = agent.act(states, add_noise=True)
            next_states, rewards, dones = env.step(actions)
            agent.step(states, actions, rewards, next_states, dones)
            score += np.mean(rewards)
            states = next_states
            if np.any(dones):
                break

        scores.append(score)
        scores_window.append(score)
        scores_avg.append(np.mean(scores_window))
        scores_std.append(np.std(scores_window))
        s_msg = '\rEpisode {}\tAverage Score: {:.2f}\tσ: {:.2f}\tScore: {:.2f}'
        print(s_msg.format(episode, np.mean(scores_window),
                           np.std(scores_window), score), end="")
        if episode % 10 == 0:
            print(s_msg.format(episode, np.mean(scores_window),
                               np.std(scores_window), score))
        if np.mean(scores_window) >= 30.:
            SOLVED = True
            s_msg = '\n\nEnvironment solved in {:d} episodes!\tAverage '
            s_msg += 'Score: {:.2f}\tσ: {:.2f}'
            print(s_msg.format(episode, np.mean(scores_window),
                               np.std(scores_window)))
            # save the models
            s_aux = '%scheckpoint-%s.%s.pth'
            s_actor_path = s_aux % (DATA_PREFIX, agent.__name__, 'actor')
            s_critic_path = s_aux % (DATA_PREFIX, agent.__name__, 'critic')
            torch.save(agent.actor_local.state_dict(), s_actor_path)
            torch.save(agent.critic_local.state_dict(), s_critic_path)
            break

    # save data to use later
    if not SOLVED:
        s_msg = '\n\nEnvironment not solved =/'
        print(s_msg.format(episode, np.mean(scores_window),
              np.std(scores_window)))
    print('\n')
    d_data = {'episodes': episode,
              'scores': scores,
              'scores_std': scores_std,
              'scores_avg': scores_avg,
              'scores_window': scores_window}
    s_aux = '%ssim-data-%s.data'
    pickle.dump(d_data, open(s_aux % (DATA_PREFIX, agent.__name__), 'wb'))
