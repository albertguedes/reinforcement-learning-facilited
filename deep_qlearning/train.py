#!/usr/bin/env python3 
# -*- coding: utf-8 -*-
#
# train.py - script to train model with deep q-reinforcement learning.
#
# created: 2021-09-21
# author: albert r. carnier guedes (albert@teko.net.br)
#
import math
import gym
import torch
import torch.nn.functional as F

# Custom modules.
from memory import Memory
from nn import NN

#
# Get an random action or from policy.
#
def get_action( env = None, Q = None, state = [], epsilon = 0.0, device = torch.device('cpu') ):

    # Get action via policy ( exploitation ) or random action ( exploration ).
    p = torch.rand(1, device=device)
    if p < epsilon:
        action = env.action_space.sample()
    else:
        # In this step, Q_pred and Q_target nn are equals. 
        # You can use any of then to find max Q.
        action = torch.argmax( Q(state,) ).item()

    return action

##########
# DEVICE #
##########
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

###############
# Environment #
###############

#   Type: Box(4)
#        Num     Observation               Min                     Max
#        0       Cart Position             -4.8                    4.8
#        1       Cart Velocity             -Inf                    Inf
#        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
#        3       Pole Angular Velocity     -Inf                    Inf
#
# Actions:
#        Type: Discrete(2)
#        Num   Action
#        0     Push cart to the left
#        1     Push cart to the right

# Select the environment.
env = gym.make('CartPole-v0')

state_dimension = env.observation_space.shape[0]
n_actions       = env.action_space.n

################
# Reply Memory #
################

# Set the memory size.
MEMORY_SIZE = 5000
# Set batch samples of experiences size.
BATCH_SIZE  = 200
# Set the reply memory.
memory = Memory(MEMORY_SIZE, 2*state_dimension + 3 )

##################
# Neural Network #
##################

n_inputs    = state_dimension
n_io_hidden = 64
n_outputs   = n_actions

# Set gradient descent learning rate.
LEARNING_RATE = 1.0e-2

# The PREDICT neural network.
Q_pred = NN(n_inputs,n_outputs,n_io_hidden).to(DEVICE)
# The TARGET neural network.
Q_target = NN(n_inputs,n_outputs,n_io_hidden).to(DEVICE)
# Clone PREDICT nn to TARGET nn.
Q_target.load_state_dict( Q_pred.state_dict() )
Q_target.eval()

# Set the optimizer of weights of the neural network.
optimizer = torch.optim.Adam( Q_pred.parameters(), lr = LEARNING_RATE)

# Select the loss function.
loss = torch.nn.SmoothL1Loss().to(DEVICE)

##############
# Q-Learning #
############## 

# Max episodes of iteration with environment. 
MAX_EPISODES = 1000
# Max steps to the agent. 
MAX_STEPS = 10*MAX_EPISODES

# Discount factor.
GAMMA = 1.0

# Exploration rate.
EPSILON_MIN = 0.1
EPSILON_MAX = 1.0
EPSILON_DECAY_FACTOR = math.pow(EPSILON_MIN/EPSILON_MAX,1.0/MAX_EPISODES)

# Init epsilon.
epsilon = EPSILON_MAX

# The frequency to learnin from reply memory.
FREQUENCY_LEARNING = 10

# Save total rewards received on one episode. 
rewards_per_episode = torch.zeros(MAX_EPISODES,dtype=torch.float, device=DEVICE)
# The loss arrived by episode.
steps_per_episode = torch.zeros(MAX_EPISODES,dtype=torch.float, device=DEVICE)

# Get the range of episodes and steps.
episodes = range(MAX_EPISODES)

for episode in episodes:

    # Get initial state.
    state = torch.tensor( env.reset() ,dtype=torch.float, device=DEVICE)

    # Not done yet.
    done = False

    # Begin the sequence of steps.
    step = 0
    while ( not done ) and ( step < MAX_STEPS ):

        # Get the action on state.
        action = get_action(env=env,Q=Q_pred,state=state,epsilon=epsilon,device=DEVICE)

        # Get next state, reward and verify final state.
        next_state, reward, done, _ = env.step(action)

        action     = torch.tensor([action],   dtype=torch.float, device=DEVICE)
        next_state = torch.tensor(next_state, dtype=torch.float, device=DEVICE)
        reward     = torch.tensor([reward],   dtype=torch.float, device=DEVICE)
        done       = torch.tensor([done],     dtype=torch.float, device=DEVICE)

        # Add the reward on reward episode.
        reward = reward - 100 * done
        rewards_per_episode[episode] = reward + GAMMA * rewards_per_episode[episode]

        # Pack all step on a one "experience".
        experience = torch.cat((state,action,reward,next_state,done))

        # Save the experience.
        memory.push(experience)

        # Update the neural networks only at a pre-defined frequency.
        if ( ( step + 1 ) % FREQUENCY_LEARNING ) == 0 :
          
            # Run minibatch if it has sufficient experience stored.
            if memory.size() > BATCH_SIZE:

                # Get a random samples of experiences.
                experiences = memory.samples(BATCH_SIZE)

                for experience in experiences:

                    state_      = experience[0:4]
                    action_     = experience[4].long()
                    reward_     = experience[5]
                    next_state_ = experience[6:10]
                    done_       = experience[10]

                    # Predict the action-value for the state and action of the experience.
                    y_pred = Q_pred(state_)[action_]

                    # Estimate the espected action-value for the state anda action of the experience.
                    # If next_state_ if last state ( done_ = 1.0 ), the only reward is from transition.
                    y_target = reward_ + ( 1 - done_ ) * GAMMA * torch.max( Q_target(next_state_) )

                    # Train the PREDICT nn.
                    optimizer.zero_grad()
                    J = loss(y_pred,y_target)
                    J.backward()
                    optimizer.step()

                # Clone PREDICT nn to TARGET nn.
                Q_target.load_state_dict( Q_pred.state_dict() )

        state = next_state

        step+= 1

    print( "Episode {:4d} .:. Step {:5d} .:. Reward {:2.2f} .:. Epsilon {:.5f}".format(
            episode, step, rewards_per_episode[episode].item(),epsilon) )

    epsilon = epsilon * EPSILON_DECAY_FACTOR
    steps_per_episode[episode] = step 

env.close()

# Save the model.
torch.save( Q_pred.state_dict(), 'saves/Q_pred.pth')

#
# PRINT RESULT
#
print("\nMean Steps:",steps_per_episode.mean().item())
print("Std  Steps:",steps_per_episode.std().item())
print("Max  Steps:",steps_per_episode.max().item())

print("\nMean Reward:",rewards_per_episode.mean().item())
print("Std  Reward:",rewards_per_episode.std().item())
print("Max  Reward:",rewards_per_episode.max().item())

exit("\nDone!\n")
