[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"



# Project 2: Continuous Control
Author: [Zhang Yu (Vorenus)](https://github.com/helsinkipirate/drlnd_vorenus)

### Introduction

For this project, I am work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of my agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

The following report is written in three parts:

- **Learning Algorithm**
- **Plot of Rewards**
- **Future Work**

## Learning Algorithm
[A2]: ../../drlnd_vorenus/p2_continuous-control/DDPG_algorithm.png "Algorithm2"

### Implimentation

As suggested by the advice of the project, I start the implementation from DDGP project. The gym environment is replaced with the `Reacher.exe` of unity environment. Starting with the DDPG implementation provided by the course, the training convergs too slowly, when I use the single node version of `Reacher.exe`. Since my training machine is not as good as before during the 'lockdown' period, I tries to work with multiple agents training. But still the multi-agent one fluctiates a lot.
According to `benchmark implementation` in `Project: Continuous Control`, the code of training is change to less aggressive with the number of updates. Also the batchsize is tuned to larger one with 1024 to help with less aggressive training. 

DDGP can be described as the following figure:
![Algorithm2][A2]


### Hyperparameters
 | Hyperparameter                      | Value |
  | ----------------------------------- | ----- |
  | Replay buffer size                  | 1e5   |
  | Batch size                          | 1024  |
  | discount factor                     | 0.99  |
  | $\tau$                              | 1e-3  |
  | LR_ACTOR                            | 1e-4  |
  | LR_CRITIC                           | 1e-3  |
  | update interval                     | 4     |
  | Number of episodes                  | 2000  |
  | Max number of timesteps per episode | 1000  |
  | L2 weight decay                     | 0     |
  | UPDATE_EVERY                        | 20    |
  | UPDATE_NETWORK                      | 10    |
  | EPSILON                             | 1.0   |
  | EPSILON_DECAY                       | 0     |
  | LEAKINESS                           | 0.01  |



### Model Architecture
------------------------------Changes needed after new results----------------------------

ACTOR NETWORK:

CRITIC NETWORK:

## Plot of Rewards

[S1]: ../../drlnd_vorenus/p2_continuous-control/Reacher_scores.png "scores"

------------------------------Changes needed after new results----------------------------
Environment solved in 425 episodes!     Average Score: 13.03
The plot of rewards is shown below.
![scores][S1]

## Future Work

In the future, some improvements can also be done: 
- Agorithms like TRPO, PPO, A3C, A2C may improve performance.