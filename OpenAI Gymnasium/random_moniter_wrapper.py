
import gymnasium
from gymnasium.wrappers import RecordVideo
if __name__=="__main__":
    env = gymnasium.make("CartPole-v1",render_mode="rgb_array")
    env = RecordVideo(env,"recoding",episode_trigger=lambda x:True)
    
    total_reward = 0.0
    total_steps = 0.0
    obs,info = env.reset()
    while True:
        action = env.action_space.sample()
        obs,reward,truncate,terminated,info = env.step(action)
        
        total_reward +=reward
        total_steps +=1
        if truncate or terminated:
            break
    print("Episode done in %d steps ,total reward %.3f" %(total_steps,total_reward))
    env.close()    
    env.env.close()
