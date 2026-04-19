from sources import P,gamma,actions
import numpy as np

def value_iterat(P,gamma,theta=1e-6):
    V = np.zeros(len(P))
    while True:
        delta = 0
        for s in P.keys():
            v = V[s]
            Q_sa = []
            for a in P[s]:
                Q_sa.append(sum(prob*(reward+gamma*V[next_s])for prob,next_s,reward in P[s][a]))
                V[s]=max(Q_sa)
                delta = max(delta,abs(v-V[s]))
        if delta<theta:
            break
    policy = np.zeros((len(P),len(actions)))
    for s in P.keys():
        Q_sa = np.zeros(len(actions))
        for a in actions:
            for prob , next_s , reward in P[s][a]:
                Q_sa[a]+=prob*(reward+gamma*V[next_s])
        best_a = np.argmax(Q_sa)
        policy[s] = np.eye(len(actions))[best_a]
    return policy,V

if __name__ =="__main__":
    optimal_policy, optimal_value = value_iterat(P, gamma)
    print("Optimal Value Function:", optimal_value)
    print("Optimal Policy:\n", optimal_policy)
