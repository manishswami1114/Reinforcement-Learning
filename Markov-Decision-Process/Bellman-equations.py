"""
->If enviroment dynamic are known
->Small MDP -> can solve system of equations.
->Otherwise->must use approximate methods
"""
from math import gamma
import numpy as np
V = np.zeros(2)

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
for _ in range(100):
    newV = np.zeros_like(V)
    for s in [0,1]:
        values=[]
        for a in [0,1,2]:
            trans = transition(s,a)
            if not trans:continue
            q=sum(p*(r+gamma*V[s_]) for p,s_,r in trans)
            values.append(s)
        newV[s]=max(values)
    V =newV
print("Optimal Values: ",V)