#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Implement a wrapper to Unity Environments


@author: udacity, ucaiado

Created on 10/07/2018
"""

import platform
from unityagents import UnityEnvironment
import yaml


'''
Begin help functions and variables
'''

PATHS = yaml.load(open('../config.yaml', 'r'))['ENVS']

'''
End help functions and variables
'''


def make():
    '''
    Return a Unity environment wrappered to works more like a OpenAI Gym Env

    :param s_parh: string. Path to the Unity environment binaries
    '''
    env = UnityEnvironment(file_name=PATHS[platform.system()])
    return EnvWrapper(env)


class EnvWrapper(object):
    '''
    Wrapper of the Unity Environment
    '''
    def __init__(self, env):
        # avoind double wrapping
        if 'Wrapper' in str(env):
            env = env.env
        self.env = env
        self.brain_name = env.brain_names[0]
        self.brain = env.brains[self.brain_name]
        self.action_size = self.brain.vector_action_space_size
        self.state_size = self.brain.vector_observation_space_size
        self.num_agents = 20

    def reset(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations

    def step(self, actions):
        env_info = self.env.step(actions)[self.brain_name]
        states = env_info.vector_observations
        rewards = env_info.rewards
        dones = env_info.local_done
        return states, rewards, dones

    def close(self):
        self.env.close()
