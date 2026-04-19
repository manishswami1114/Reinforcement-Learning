import numpy as np
import torch

class BanditEnv:
    def __init__(self, k=10):
        self.q_true = np.random.normal(0, 1, k)  # true action values
        self.k = k

    def step(self, action):
        reward = np.random.normal(self.q_true[action], 1.0)
        return reward

class UCBAgent:
    def __init__(self, k=10, c=2.0):
        self.k = k
        self.c = c
        self.Q = torch.zeros(k)         # estimated values
        self.N = torch.zeros(k)         # action counts
        self.t = 0
    
    def select_action(self):
        self.t += 1
        # force selection of untried actions
        for a in range(self.k):
            if self.N[a] == 0:
                return a
        # compute UCB values
        ucb_values = self.Q + self.c * torch.sqrt(torch.log(torch.tensor(self.t)) / self.N)
        return torch.argmax(ucb_values).item()
    
    def update(self, action, reward):
        self.N[action] += 1
        # incremental average update
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

# Run experiment
np.random.seed(1)
env = BanditEnv(k=10)
agent = UCBAgent(k=10, c=2.0)

rewards = []
for step in range(100000):
    action = agent.select_action()
    reward = env.step(action)
    agent.update(action, reward)
    rewards.append(reward)

print("Average reward over 1000 steps:", np.mean(rewards))
print("Final estimated Q-values:", agent.Q)
print("True Q-values:", env.q_true)
