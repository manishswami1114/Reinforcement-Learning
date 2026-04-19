import numpy as np
import torch

class BanditEnv:
    def __init__(self,k=10):
        self.q_true= np.random.normal(4,1,k)
        self.k=k
    def step(self,action):
        reward =np.random.normal(self.q_true[action],1.0)
        return reward
class GradientBanditAgent:
    def __init__(self,k=10,alpha=0.01,use_baseline=True):
        self.k=k
        self.alpha = alpha
        self.H = torch.zeros(k,dtype=torch.float32)
        self.pi=torch.ones(k)/k
        self.use_baseline=use_baseline
        self.avg_reward=0.0
        self.t= 0
    def select_action(self):
        probs = torch.softmax(self.H,dim=0)
        self.pi=probs
        return torch.multinomial(probs,1).item()
    def update(self,action,reward):
        self.t +=1
        if self.use_baseline:
            self.avg_reward+=(reward-self.avg_reward)/self.t
        baseline=self.avg_reward if self.use_baseline else 0.0
        one_hot = torch.zeros(self.k)
        one_hot[action]=1.0
        
        self.H +=self.alpha *(reward-baseline)*(one_hot-self.pi)
    
    
# Run experiment
np.random.seed(42)
env = BanditEnv(k=10)
agent = GradientBanditAgent(k=10,alpha=0.1,use_baseline=True)
rewards =[]
steps=1000
for step in range(steps):
    action = agent.select_action()
    reward = env.step(action)
    agent.update(action,reward)
    rewards.append(reward)
print(f"Average reward over {steps} steps: ",np.mean(rewards))
print("Final action probabilities: ",agent.pi)
print("True Q-Values: ",env.q_true)