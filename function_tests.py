import numpy as np

class MockEnv:
    def __init__(self):
        self.current_state = 0
        self.step_count = 0

    def reset(self):
        self.current_state = 0
        self.step_count = 0
        return self.current_state

    def step(self, action):
        # Simulate the environment's response to an action
        self.current_state += 1
        reward = 1  # Fixed reward for simplicity
        self.step_count += 1
        done = self.step_count >= 10  # Terminal state after 10 steps
        trunc = False  # For simplicity, we won't use truncation in this mock
        info = {}
        return self.current_state, reward, done, trunc, info


def test_skip_frame():
    env = MockEnv()
    skip_env = SkipFrame(env, skip=4)
    total_rewards = []
    done = False

    while not done:
        next_state, total_reward, done, trunc, info = skip_env.step(action=0)
        total_rewards.append(total_reward)
        print(f"Next State: {next_state}, Total Reward: {total_reward}, Done: {done}")

    # Check if the total rewards accumulated are as expected
    expected_rewards = [4] * (10 // 4)  # Adjust based on your mock environment and skip value
    assert all(tr == er for tr, er in zip(total_rewards, expected_rewards)), "Test failed: Rewards do not match expected values."

    print("Test passed: Step function works as expected.")
