[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e.gif "Agent"



# Project 1: Navigation

Author: [Zhang Yu (Vorenus)](https://github.com/helsinkipirate/drlnd_vorenus)

This project demonstrates the ability of value-based methods, specifically, [Deep Q-learning](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) and its variants, to learn a suitable policy in a model-free Reinforcement Learning setting using Unity environment, which consists of a continuous state space of 37 dimensions, with the goal to navigate around and collect yellow bananas (reward: +1) while avoiding blue bananas (reward: -1). There are 4 actions to choose from: move left, move right, move forward and move backward. An agent acting can be seen as below:

![Agent][image1]

The following report is written in three parts:

- **Learning Algorithm**
- **Plot of Rewards**
- **Future Work**

## Learning Algorithm
[A1]: ../../drlnd_vorenus/p1_navigation/DQN_algorithm.png "Algorithm1"

### Implimentation
The learning algorithm Deep Q-network is employed by the project. As a temporal-difference learning (TD-learning), deep Q-nework is to learn a action-value function, which is used to optimize the estimated future rewards. The algoirthm can be described as the following figure
![Algorithm1][A1]

### Hyperparameters
 | Hyperparameter                      | Value |
  | ----------------------------------- | ----- |
  | Replay buffer size                  | 1e5   |
  | Batch size                          | 64    |
  | discount factor                     | 0.99  |
  | $\tau$                              | 1e-3  |
  | LR, Learning rate                   | 5e-4  |
  | update interval                     | 4     |
  | Number of episodes                  | 500   |
  | Max number of timesteps per episode | 3000  |
  | Epsilon start (greedy policy)       | 1.0   |
  | Epsilon minimum (greedy policy)     | 0.01  |
  | Epsilon decay (greedy policy)       | 0.995 |


### Model Architecture

Our DQN employs three NN layers as the following:
`fc1 = nn.Linear(state_size, fc1_units)`
`fc2 = nn.Linear(fc1_units, fc2_units)`
`fc3 = nn.Linear(fc2_units, action_size)`
The first two layers are activated with RELU functions.

## Plot of Rewards

[R1]: ../../drlnd_vorenus/p1_navigation/banana_game_rewards.png "rewards"

Environment solved in 425 episodes!     Average Score: 13.03
The plot of rewards is shown below.
![rewards][R1]

## Future Work

In the future, some improvements can also be done: 
-the DQN can also be replaced with double DQN, Dueling DNQ.
-Using prioritized replay



