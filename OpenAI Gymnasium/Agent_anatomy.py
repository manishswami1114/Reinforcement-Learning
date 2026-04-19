import random
from typing import List 

class Enviroment:
    def __init__(self):
        self.step_left = 10
    def get_observation(self)->List[float]:
        return [0.0,0.0,0.0]
    def get_action(self)-> List[int]:
        return[0,1]
    def is_done(self)->bool:
        return self.step_left==0
    def action(self,action:int)-> float:
        if self.is_done():
            raise Exception("game is over")
        self.step_left-=1
        return random.random()

class Agent():
    def __init__(self):
        self.total_reward =0.0
    def step(self,env:Enviroment):
        current_obs = env.get_observation()
        actions = env.get_action()
        reward = env.action(random.choice(actions))
        self.total_reward+=reward

if __name__=="__main__":
    env = Enviroment()
    agent = Agent()
    while not env.is_done():
        agent.step(env)
    print("Total reward got: %.4f"% agent.total_reward)