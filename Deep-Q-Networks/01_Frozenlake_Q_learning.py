import gymnasium
import collections
from tensorboardX import SummaryWriter

ENV_NAME = "FrozenLake-v1"

GAMMA = 0.9
ALPHA = 0.2  # learning rate
TEST_EPISODES = 20


def normalize_state(s):
    """
    Return a hashable representation of the observation.
    - If s is a dict -> convert to tuple of (key, value) sorted (assuming values are hashable)
    - If s is a numpy array -> convert to tuple(s.tolist())
    - Otherwise return s (if already hashable like int)
    """
    # Lazy import to avoid dependency when not needed
    try:
        import numpy as np
    except Exception:
        np = None

    if isinstance(s, dict):
        # convert dict to tuple of items sorted by key to get deterministic order
        return tuple((k, normalize_state(v)) for k, v in sorted(s.items()))
    if np is not None and isinstance(s, np.ndarray):
        return tuple(s.tolist())
    if isinstance(s, (list, tuple)):
        # recursively normalize list/tuple
        return tuple(normalize_state(x) for x in s)
    # primitive (int, float, str) assumed hashable
    return s


class Agent:
    def __init__(self):
        self.env = gymnasium.make(ENV_NAME)
        obs, _ = self.env.reset()
        self.state = normalize_state(obs)
        self.values = collections.defaultdict(float)

    def sample_env(self):
        action = self.env.action_space.sample()
        old_state = self.state
        new_obs, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated
        if done:
            new_reset_obs, _ = self.env.reset()
            self.state = normalize_state(new_reset_obs)
        else:
            self.state = normalize_state(new_obs)
        return old_state, action, reward, normalize_state(new_obs)

    def best_value_and_action(self, state):
        best_value, best_action = None, None
        for action in range(self.env.action_space.n):
            action_value = self.values[(state, action)]
            if best_value is None or best_value < action_value:
                best_value = action_value
                best_action = action
        return best_value, best_action

    def value_update(self, s, a, r, next_s):
        # If next_s has never been seen, best_value will be 0.0 (defaultdict)
        best_v, _ = self.best_value_and_action(next_s)
        new_v = r + GAMMA * best_v
        old_v = self.values[(s, a)]
        self.values[(s, a)] = old_v * (1 - ALPHA) + new_v * ALPHA

    def play_episode(self, env):
        total_reward = 0.0
        obs, _ = env.reset()
        state = normalize_state(obs)
        while True:
            _, action = self.best_value_and_action(state)
            # if action is None (no info yet), sample a random action
            if action is None:
                action = env.action_space.sample()
            new_obs, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            done = terminated or truncated
            if done:
                break
            state = normalize_state(new_obs)
        return total_reward


if __name__ == "__main__":
    test_env = gymnasium.make(ENV_NAME)
    agent = Agent()
    writer = SummaryWriter(comment="Q-learning")

    iter_no = 0
    best_reward = 0.0
    while True:
        iter_no += 1
        s, a, r, next_s = agent.sample_env()
        # Debug prints (uncomment if you want to inspect types)
        # print(type(s), s, type(next_s), next_s)
        agent.value_update(s, a, r, next_s)

        reward = 0.0
        for _ in range(TEST_EPISODES):
            reward += agent.play_episode(test_env)
        reward /= TEST_EPISODES

        writer.add_scalar('reward', reward, iter_no)

        if reward > best_reward:
            print("Best reward updated %.3f -> %.3f" % (best_reward, reward))
            best_reward = reward

        if reward > 0.80:
            print("Solved in %d iterations!" % iter_no)
            break

    writer.close()
