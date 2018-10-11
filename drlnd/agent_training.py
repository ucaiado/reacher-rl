from ddpg_agent import MetaAgent
from collections import deque
from unityagents import UnityEnvironment
import numpy as np
import platform
import pdb

d_path = {'Linux': '../Reacher.app/',
          'Windows': 'C:/GIT/reacher-rl/Reacher_Windows_x86_64/Reacher.exe',
          'Darwin': '../Reacher.app'}

if __name__ == '__main__':

    env = UnityEnvironment(file_name=d_path[platform.system()])

    # get the default brain
    print('\n')
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    s_msg = 'There are {} agents. Each observes a state with length: {}'
    print(s_msg.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])

    # from drlnd.ddpg_agent import MetaAgent
    nb_agents = 20
    episodes = 10
    rand_seed = 0

    scores_list = []
    scores_window = deque(maxlen=100)  # last 100 scores

    agent = MetaAgent(state_size, action_size, nb_agents, rand_seed)

    for episode in range(episodes):
        env_info = env.reset(train_mode=True)[brain_name]
        # Reset the enviroment
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        # while True:
        for i in range(1000):
            # Predict the best action for the current state.
            actions = agent.act(states, add_noise=True)
            actions = np.clip(actions, -1, 1)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            agent.step(states, actions, rewards, next_states, dones)
            scores += env_info.rewards
            states = next_states
            if np.any(dones):
                break

        # pdb.set_trace()
        scores_window.append(scores)
        scores_list.append(scores)
        s_msg = '\rEpisode {}\tAverage Score: {:.2f}'
        print(s_msg.format(episode, np.mean(scores_window)), end="")
        if episode % 10 == 0:
            s_msg = '\rEpisode {}\tAverage Score: {:.2f}'
            print(s_msg.format(episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 30.0:
            s_msg = '\nEnvironment solved in {:d} episodes!\tAverage '
            s_msg += 'Score: {:.2f}'
            print(s_msg.format(episode, np.mean(scores_window)))
            break
