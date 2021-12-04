# environment.py - simulate a environment with discrete states, that receives a discrete
#                  action and an sttate and return a new state and a reward.
#
# created: 2021-09-16
# author: albert r. carnier guedes (albert@teko.net.br)
#

import torch

# Class to represent the environment.
class Environment:

    # A tensor with N_STATES. Each state is represented by a integer number.
    _states_table = None

    # A tensor with N_STATES X N_ACTIONS X N_STATES. Rewards are any type. 
    _rewards_table = None

    # A tensor with N_STATES X N_ACTIONS. Returna integer that 
    # represent the state returned when given an state and a action. 
    _transitions_table = None

    def __init__( self, states_table = None, rewards_table = None, transitions_table = None ):
        self._states_table       = states_table
        self._rewards_table      = rewards_table
        self._transitions_table = transitions_table

    # Set new states.
    def set_states(self, states_table = None ):
        self._states_table = states_table

    # Get environment states.
    def get_states(self):
        return self._states_table

    # Set rewards.
    def set_rewards(self, rewards_table = None ):
        self._rewards_table = rewards_table

    # Get rewards.
    def get_rewards(self):
        return self._rewards_table

   # Set the transitions.
    def set_transitions(self, transitions_table = None ):
        self._transitions_table = transitions_table

    # Get environment transitions.
    def get_transitions(self):
        return self._transitions_table

    # Give a state and a action, return a new state.
    def transition(self, state = None , action = None ):
        return self._transitions_table[state, action]

    # Return the reward by executing the action on a given state and arrival a 
    # new state.
    def reward(self, state = None, action = None, new_state = None ):
        return self._rewards_table[state,action,new_state]

    # Return possible actions to be executed on a given state.
    def actions(self, state = None ):
        actions, = torch.where( self._transitions_table[state,] > -1 )
        return actions
