import sys
import numpy as np
sys.path.append("../")

try:
    from models import Actor, Critic, Policy
    from utils import param_table
    from utils.make_env import make
except ModuleNotFoundError:
    from .models import Actor, Critic, Policy
    from drlnd.utils import param_table
    from drlnd.utils.make_env import make


class Trajectory(object):
    def __init__(self):
        self.data = {}
        self.data['state'] = []
        self.data['reward'] = []
        self.data['prob'] = []
        self.data['action'] = []
        self.data['value'] = []
        self.score = 0.

    def add(self, states, rewards, probs, actions, values):
        iter_obj = zip(states, rewards, probs, actions, values)
        self.data['state'].append(states)
        self.data['reward'].append(rewards)
        self.data['prob'].append(probs)
        self.data['action'].append(actions)
        self.data['value'].append(values)
        self.score += np.mean(rewards)
        # for state, reward, prob, action, value in iter_obj:
        #     self.data['state'].append(state)
        #     self.data['reward'].append(reward)
        #     self.data['prob'].append(prob)
        #     self.data['action'].append(action)
        #     self.data['value'].append(value)
        #     self.score += np.mean(reward)

    def __len__(self):
        return len(self.state_list)

    def __getitem__(self, key):
        return self.data[key]
