import numpy as np 
import torch 

n_actions = 10
q_true = np.random.randn(n_actions) # true values
q_est = torch.zeros(n_actions) # Q_t(a),start with 0
N = torch.zeros(n_actions) # count of actions taken

epsilon = 0.1
n_steps = 1000
rewards=[]
for t in range(1,n_steps+1):
    #  e-greedy action selection 
    if np.random.rand()<epsilon:
        action = np.random.randint(n_actions) # explore
    else:
        action = torch.argmax(q_est).item() # exploit
    
    # <---Generate reward from enviroment---> 
    reward = np.random.randn()+q_true[action] # true value +noice
    
    N[action]+=1 # update counts
    
    # <--- Sample-average update --->
    q_est[action] += (reward-q_est[action]/N[action])
    
    rewards.append(reward)
print("final estimated values:",q_est)
print("True values:",q_true)
print("Average reward:",np.mean(rewards))