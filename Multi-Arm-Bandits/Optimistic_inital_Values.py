import torch
import torch.nn as nn
import numpy as np

class BanditEnv:
    def __init__(self, k=10):
        # true action values ~ N(0,1)
        self.q_true = np.random.normal(0, 1, k)
        self.k = k
    
    def step(self, action):
        # reward = true value + noise
        reward = np.random.normal(self.q_true[action], 1.0)
        return reward

class OptimisticGreedyAgent:
    def __init__(self, k=10, alpha=0.1, optimistic_init=5.0):
        self.k = k
        self.alpha = alpha
        # Optimistic initial values
        self.Q = torch.ones(k) * optimistic_init
        self.action_counts = torch.zeros(k)
    
    def select_action(self):
        # purely greedy
        return torch.argmax(self.Q).item()
    
    def update(self, action, reward):
        self.action_counts[action] += 1
        # incremental update with constant α
        self.Q[action] += self.alpha * (reward - self.Q[action])

# Run experiment
np.random.seed(0)
env = BanditEnv(k=10)
agent = OptimisticGreedyAgent(k=10, alpha=0.1, optimistic_init=5.0)

rewards = []
for t in range(100000):
    action = agent.select_action()
    reward = env.step(action)
    agent.update(action, reward)
    rewards.append(reward)

print("Average reward over 1000 steps:", np.mean(rewards))
print("Final estimated Q-values:", agent.Q)
print("True Q-values:", env.q_true)
