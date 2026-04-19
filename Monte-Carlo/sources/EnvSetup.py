import numpy as np 

np.random.seed(42)

n_states = 3 
n_actions = 2 
gamma = 0.9 

def b_policy(s):
    return np.array([0.5,0.5])
def pi_policy(s):
    probs = np.array([0.1,0.9])
    return probs/probs.sum()

def reward_fn(s,a):
    return np.random.normal(1+s+0.5*a,0.1)
def generate_episode(policy_fn , max_steps=5):
    s = np.random.randint(0,n_states)
    episode=[]
    for t in range(max_steps):
        probs = policy_fn(s)
        a = np.random.choice(n_actions,p=probs)
        r = reward_fn(s,a)
        episode.append((s,a,r))
        s = (s+a+1)%n_states
    return episode
        