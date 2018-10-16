#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
The __init__.py files are required to make Python treat the directories as
containing packages; this is done to prevent directories with a common name,
such as string, from unintentionally hiding valid modules that occur later
(deeper) on the module search path.


@author: ucaiado

Created on 10/06/2018
"""

from .utils.make_env import make
from .ddpg.agent import Agent as DDPG
from .ddpg.agent import set_global_parms
from .ddpg.agent import PARAMS as DDPG_PARAMS
from .ppo.agent import Agent as PPO
from .ppo.agent import set_global_parms
from .ppo.agent import PARAMS as PPO_PARAMS
