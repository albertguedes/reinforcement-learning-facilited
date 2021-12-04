# environment-0.py - a single environment with discrete states and actions.
#
# created: 2021-09-16
# author: albert r. carnier guedes (albert@teko.net.br)
#

import torch 

# Set the number of states.
N_STATES = torch.tensor(4,dtype=torch.long)
# Set the states.
STATES = torch.tensor(range(N_STATES),dtype=torch.long)

# Set the number of actions. Set action = 0 for no-action, or, stay on same state.
N_ACTIONS = torch.tensor(2,dtype=torch.long)
# Set the actions.
ACTIONS = torch.tensor(range(N_ACTIONS),dtype=torch.long)

# NOW, the transition table is a PROBABILITY TABLE of transition from a state S to 
# S', executing an action A on S.
TRANSITIONS = torch.zeros([N_STATES,N_ACTIONS,N_STATES],dtype=torch.float)
# Set the TRANSITIONS PROBABILITIES.
# For each pair (S,A), the sum of probabilities sum_S' (T(S,A,S')) is equal o 1.
TRANSITIONS[0,0,1] = torch.tensor(1.0,dtype=torch.float)
TRANSITIONS[0,1,2] = torch.tensor(1.0,dtype=torch.float)
TRANSITIONS[1,0,3] = torch.tensor(1.0,dtype=torch.float)
TRANSITIONS[2,0,1] = torch.tensor(1.0,dtype=torch.float)
TRANSITIONS[2,1,3] = torch.tensor(1.0,dtype=torch.float)

# Reward for starting from state S, performing action a, and arriving in state s':
# R = REWARDS(S,A,S')
# Set all the REWARDS as -1 by default.
REWARDS = torch.zeros([N_STATES,N_ACTIONS,N_STATES],dtype=torch.float)
# Set the REWARDS.
REWARDS[0,0,1] = torch.tensor(1.0,dtype=torch.float)
REWARDS[0,1,2] = torch.tensor(2.0,dtype=torch.float)
REWARDS[1,0,3] = torch.tensor(4.0,dtype=torch.float)
REWARDS[2,0,1] = torch.tensor(3.0,dtype=torch.float)
REWARDS[2,1,3] = torch.tensor(5.0,dtype=torch.float)
