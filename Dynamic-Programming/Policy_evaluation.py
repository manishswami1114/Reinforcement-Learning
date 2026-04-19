from sources import P, actions,gamma
import numpy as np

def policy_eval(policy, P , gamma=0.9,theta=1e-6):
    V = np.zeros(len(P))
    while True:
        delta = 0
        for s in P.keys():
            v=  0
            for a ,action_prob in enumerate(policy[s]):
                for prob, next_s ,reward in P[s][a]:
                    v+=action_prob*prob*(reward+gamma*V[next_s])
            delta=max(delta,abs(V[s]-v))
            V[s]=v
        if delta < theta:
            break
    return V

policy = np.ones((len(P),len(actions)))/len(actions)

V = policy_eval(policy,P,gamma)
print("State values under current Policy:\n",V)