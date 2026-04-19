import numpy as np 
import torch 
n_actions =10
q_true = np.random.randn(n_actions)
q_est= torch.zeros(n_actions)

alpha = 0.1
epsilon = 0.1
n_steps= 1000
rewards=[]

for  i in range(n_steps):
    q_true +=np.random.normal(0,0.01,size=n_actions)
    
    if np.random.rand()<epsilon:
        action = np.random.randint(n_actions)
    else:
        action = torch.argmax(q_est).item()
    reward = np.random.randn()+q_true[action]
    
    q_est[action]+=alpha*(reward-q_est[action])
    rewards.append(reward)
print("Estimated_values:",q_est)
print("True Values:",q_true)
print("Average reward:",np.mean(rewards))