#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 14 18:16:13 2018

@author: feebr01
"""

import numpy as np
import pandas as pd


matrix = pd.DataFrame({'sleep':[.2, .1, .2], 'run':[.6, .6, .7], 'ice_cream':[.2, .3, .1]}, 
              index = ['sleep', 'run', 'ice_cream'])


# initial state of model
initial_state = 'run'
state_seq = [initial_state]


# iterations in model
for i in range(1,100):
    prior_state = state_seq[i-1]
    probs = matrix.loc[prior_state]
    choices = list(probs.index)
    choice_probs = list(probs.values)
    current_state = np.random.choice(choices, p = choice_probs)
    state_seq.append(str(current_state))
    current_state = prior_state
    

# create percentage of time in each state using counter on whole list
from collections import Counter as C
from collections import defaultdict
state_counts = C(state_seq)
tot_states = sum(C(state_seq).values())
states_pct = defaultdict(list)


# dictionary of percentage of days by state
for k,v in state_counts.items():
    states_pct[k] = v/tot_states
    
