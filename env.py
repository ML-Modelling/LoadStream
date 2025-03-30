import gym
from gym import spaces
import numpy as np


class LoadBalancer(gym.Env):
    def __init__(self):
        super(LoadBalancer, self).__init__()

        self.action_space = spaces.MultiDiscrete([11, 11, 11, 21])
        self.observation_space = spaces.Box(low=np.array([0, 0], dtype=np.float32),
                                            high=np.array(
                                                [20, 20], dtype=np.float32),
                                            dtype=np.float32)

        self.state = np.array([10, 10], dtype=np.float32)
        self.time_step = 0

        # Performance metrics
        self.demand_history = []
        self.storage_history = []
        self.rewards_history = []
        self.latency_history = []
        self.throughput_history = []
        self.response_time_history = []
        self.efficiency_history = []
        self.completion_rate_history = []
        self.time_steps_history = []

    def step(self, action):
        demand_pattern = 10 + 3 * \
            np.sin(self.time_step * 0.3 * np.pi) + 2 * \
            np.sin(self.time_step * 0.1 * np.pi)
        self.state[0] = demand_pattern + np.random.normal(0, 1.5)
        self.time_step += 1

        total_production = sum(action[:-1])
        storage_action = action[-1]

        self.state[1] += total_production - self.state[0]
        self.state[1] = min(max(self.state[1], 0), 20)

        reward = -abs(total_production -
                      self.state[0]) - abs(storage_action - self.state[1])

        # Calculate performance metrics
        latency = np.random.uniform(5, 15)
        throughput = total_production / (latency + 0.1)
        response_time = latency + np.random.uniform(0.1, 1.0)
        efficiency = (total_production / max(self.state[0], 1)) * 100
        completion_rate = 1 if total_production >= self.state[0] else 0
        self.demand_history.append(self.state[0])
        

        self.demand_history.append(self.state[0])
        self.storage_history.append(self.state[1])
        self.rewards_history.append(reward)
        self.latency_history.append(latency)
        self.throughput_history.append(throughput)
        self.response_time_history.append(response_time)
        self.efficiency_history.append(efficiency)
        self.completion_rate_history.append(completion_rate)
        self.time_steps_history.append(self.time_step)

        done = False
        info = {}

        return self.state, reward, done, info

    def reset(self):
        self.state = np.array([10, 10], dtype=np.float32)
        self.time_step = 0
        return self.state

    def render(self, mode='console'):
        if mode != 'console':
            raise NotImplementedError()
        print(
            f"Step: {self.time_step}, Demand: {self.state[0]}, Storage: {self.state[1]}")

    def seed(self, seed=None):
        np.random.seed(seed)