import gymnasium

from typing import TypeVar
import random
Action = TypeVar('Action')

class RandomActionWrapper(gymnasium.ActionWrapper):
    def __init__(self, env,epsilon=0.1):
        super(RandomActionWrapper,self).__init__(env)
        self.epsilon = epsilon
    def action(self,action:Action)->Action:
        if random.random()<self.epsilon:
            print("Random!")
            return self.env.action_space.sample()
        return action

if __name__=="__main__":
    env = RandomActionWrapper(gymnasium.make("CartPole-v1"))
    
    obs = env.reset()
    total_reward = 0.0
    
    while True:
        obs, reward,truncate,terminated,info = env.step(0)
        total_reward +=float(reward)
        if truncate or terminated:
            break
        print("Reward got: %.2f" %total_reward)