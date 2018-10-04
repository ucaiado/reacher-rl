Reacher RL
===========================


In this project, I will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment that contains 20 identical agents, each with its own copy of the environment. 

<p align="center"><img src="https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif" alt="Example" width="50%" style="middle"></p>

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

To solve the environment, the agents must get an average score of +30 over 100 consecutive episodes, and over all agents. This project is part of the [Deep Reinforcement Learning Nanodegree](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwigwuKwr4LdAhUMI5AKHTuBCz0QFjAAegQIDBAB&url=https%3A%2F%2Fwww.udacity.com%2Fcourse%2Fdeep-reinforcement-learning-nanodegree--nd893&usg=AOvVaw3OfEe4LlR9h_4vW3TZpE_o) program, from Udacity. You can check my report [here](reports/Report.pdf).


### Install
This project requires **Python 3.5** or higher, the Reacher Collector Environment (follow the instructions to download [here](drlnd/README.md)) and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Torch](https://pytorch.org)
- [UnityAgents](https://github.com/Unity-Technologies/ml-agents)


### Run
In a terminal or command window, navigate to the top-level project directory `reacher-rl/` (that contains this README) and run the following command:

```shell
$ jupyter notebook
```

This will open the Jupyter Notebook software and notebook in your browser which you can use to explore and reproduce the experiments that have been run. 


### References
1. Bla


### License
The contents of this repository are covered under the [MIT License](LICENSE).
