"""
Condition to use:-
->Enviroment satisfies Markov Property (future depends only on current state,not history)
->State space & action space are finite
->Needed when formalizing RL tasks
"""

import numpy as np
# State: h=0(high),l=1(low)
# Actions: search=0,wait=1,recharge=2
alpha,beta= 0.8,0.6
r_search,r_wait=1,0.5
gamma = 0.9

def transition(s,a):
    if s==0:
        if a==0: return [(alpha,0,r_search),(1-alpha,1,r_search)]
        if a==1: return [(1.0,0,r_wait)]
    if s==1:
        if a==0: return [(beta,1,r_search),(1-beta,0,-3)]
        if a==1: return [(1.0,1,r_wait)]
        if a==2: return [(1.0,0,0)]
    return []
print("Transitions from (low ,search): ",transition(1,0))