[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135623-e770e354-7d12-11e8-998d-29fc74429ca2.gif "Trained Agent"


# Project 3: Collaboration and Competition
Author: [Zhang Yu (Vorenus)](https://github.com/helsinkipirate/drlnd_vorenus)

### Introduction

For this project, you will work with the [Tennis](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#tennis) environment.

![Trained Agent][image1]

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1.  If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01.  Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation.  Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
- This yields a single **score** for each episode.

The environment is considered solved, when the average (over 100 episodes) of those **scores** is at least +0.5.

The following report is written in three parts:

- **Learning Algorithm**
- **Plot of Rewards**
- **Future Work**

## Learning Algorithm
[A2]: DDPG_algorithm.png "Algorithm2"

### Implimentation

Still, this project is implemented and started from DDPG project, with different tennis unity enviroment, rather than a gym environment. The difference is that `Tennis_solution.ipynb` requires multi-agent support. A lib supporting multiple agents is implented to generate multiple ddpg agents, besides a purely `ddpg_agent` lib. The operations of replay buffer is also changed into `ma_agent.py`, since both agents are using the same replay buffer. The shared replay buffer helps to ensure all the agents explore all the environment. The learning operation here is also called in the multiple agent library. When tuning, the batch size is set to 128 is enough for training.    

Due to the "lockdown" of the city, I have no gpu around. The codes are running on different cpu machines, as well as the online workplace of udacity. The first one achieving the required score, is submitted to the github. The running jupyter notebook is 'Tennis_solution.ipynb'. It takes nearly 4800 episodes. 

Still, if the environment in the current folder cannot be initialized, new download according to the instruction in the readme.md is necessary. It happens when training the same codes on different machines.


### Hyperparameters
| Hyperparameter                      | Value |
| ----------------------------------- | ----- |
| Replay buffer size                  | 10e6  |
| Batch size                          | 128   |
| $\gamma$ (discount factor)          | 0.99  |
| $\tau$                              | 1e-3  |
| Actor Learning rate                 | 1e-4  |
| Critic Learning rate                | 1e-3  |
| Weight update frequency             | 2     |
| L2 weight decay                     | 0.0   |
| Noise amplification                 | 1     |
| Noise amplification decay           | 1     |



### Model Architecture

Critic network is a bit different from the previous contiuous control project.

ACTOR NETWORK:
- self.fc1 = nn.Linear(state_size, fc1_units)
- self.fc2 = nn.Linear(fc1_units, fc2_units)
- self.fc3 = nn.Linear(fc2_units, action_size)


CRITIC NETWORK:

- self.fcs1 = nn.Linear((state_size+action_size)*n_agents, fcs1_units)
- self.fc2 = nn.Linear(fcs1_units, fc2_units)
- self.fc3 = nn.Linear(fc2_units, 1)


## Plot of Rewards

[S1]: Scores_tennis.png "scores"

--------------------------------------------------------
*** Environment solved in 4731 episodes!	Average Score: 0.5091 ***
The plot of rewards is shown below.
![scores][S1]

## Future Work

In the future, some improvements can also be done: 
-  Using Prioritized Replay ([paper](https://arxiv.org/abs/1511.05952)) has generally shown to have been quite useful. It is expected that it'll lead to an improved performance here too.

- Other algorithms like TRPO, PPO, A3C, A2C also have potential to improve the performance.
