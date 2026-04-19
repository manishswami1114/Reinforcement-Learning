from sources import P,gamma,actions,states
from Policy_evaluation import V
import numpy as np

def policy_improve(V,P,gamma=0.9):
    policy = np.zeros((len(P),len(actions)))
    for s in P.keys():
        Q_sa = np.zeros(len(actions))
        for a in actions:
            for prob,next_s,reward in P[s][a]:
                Q_sa[a]+=prob*(reward+gamma*V[next_s])
        best_action = np.argmax(Q_sa)
        policy[s]=np.eye(len(actions))[best_action]
    return policy

if __name__=="__main__":
    improved_policy = policy_improve(V,P,gamma)
    print("Improved deterministic policy:\n",improved_policy)
    