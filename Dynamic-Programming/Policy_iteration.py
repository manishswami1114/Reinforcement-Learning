from sources import P,gamma,states,actions
import numpy as np
from Policy_evaluation import policy_eval
from Policy_improvement import policy_improve

def policy_iterat(P,gamma=0.9,theta=1e-6):
    policy = np.ones((len(P),len(actions)))/len(actions)
    
    while True:
        V= policy_eval(policy,P,gamma,theta)
        new_policy = policy_improve(V,P,gamma)
        
        if np.array_equal(new_policy,policy):
            break
        policy= new_policy
        
    return policy,V

if __name__=="__main__":
    optimal_policy, optimal_value = policy_iterat(P,gamma)
    print("Optimal Policy:\n",optimal_policy)
    print("Optimal state Values:\n",optimal_value)