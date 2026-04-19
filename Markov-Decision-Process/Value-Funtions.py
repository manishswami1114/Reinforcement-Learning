"""
Conditions to use:-
->When you want to evaluate a given policy
->Works only if you know transitions or can sample enough experience
"""
import numpy as np
V = np.zeros(2)
policy = {0:0,1:2} # high->search, low -> recharge
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
for _ in range(50):
    newV=np.zeros_like(V)
    for s in [0,1]:
        a = policy[s]
        for p,s_next,r in transition(s,a):
            newV[s]+=p*(r+gamma*V[s_next])
    V = newV
print("Value of states under policy: ",V)