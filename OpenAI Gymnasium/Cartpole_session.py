import gymnasium

if __name__ == "__main__":
    env = gymnasium.make("CartPole-v1")
    total_reward = 0.0
    total_steps = 0
    obs, info = env.reset()

    while True:
        action = env.action_space.sample()
        obs, reward,terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_steps += 1
        if terminated or truncated:
            break
    print("episode done in %d steps , total reward %.2f" % (total_steps, total_reward))
