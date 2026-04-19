from sources import P,gamma,actions
import numpy as np
def policy_iteration(P,gamma,theta=1e-6):
    Q = np.zeros((len(P),len(actions)))
    
    while True:
        delta= 0
        for s in P.keys():
            for a in actions:
                old_q = Q[s,a]
                Q[s,a]=sum(
                    prob*(reward+gamma*np.max(Q[next_s]))
                    for prob,next_s,reward in P[s][a]
                )
                delta = max(delta,abs(old_q-Q[s,a]))
        if delta<theta:
            break
        
    policy = np.zeros((len(P),len(actions)))
    for s in P.keys():
        best_action = np.argmax(Q[s])
        policy[s]=np.eye(len(actions))[best_action]
    return policy,Q

if __name__=="__main__":
    optimal_policy, optimal_value = policy_iteration(P, gamma)
    print("✅ Optimal Policy:\n", optimal_policy)
    print("🏆 Optimal State Values:\n", optimal_value)
