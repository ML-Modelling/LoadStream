import numpy as np
import random

class LoadBalancer:
    def __init__(self, num_servers=5):
        self.num_servers = num_servers
        self.time_steps_history = []
        self.demand_history = []
        self.storage_history = []
        self.rewards_history = []
        self.reset()

    def reset(self):
        self.servers = [{'load': 0, 'capacity': random.randint(10, 50), 'latency': random.randint(50, 300),
                         'throughput': random.randint(500, 1000), 'health': random.randint(50, 100),
                         'response_time': random.randint(10, 100), 'failure_rate': random.uniform(0, 0.1)}
                        for _ in range(self.num_servers)]
        self.time_step = 0
        self.demand = random.randint(50, 200)
        self.storage = random.randint(20, 100)
        self.time_steps_history.clear()
        self.demand_history.clear()
        self.storage_history.clear()
        self.rewards_history.clear()
        return self.get_observation()

    def get_observation(self):
        return np.array([server['load'] / max(1, server['capacity']) for server in self.servers])

    def reward_function(self, server):
        load_penalty = -server['load'] / max(1, server['capacity'])
        latency_score = -server['latency'] / 300
        throughput_score = server['throughput'] / 1000
        health_penalty = -(100 - server['health']) / 100
        response_penalty = -server['response_time'] / 100
        failure_penalty = -server['failure_rate']
        return load_penalty + latency_score + throughput_score + health_penalty + response_penalty + failure_penalty

    def step(self, action):
        selected_server = self.servers[action]
        selected_server['load'] += random.randint(5, 20)
        selected_server['health'] = max(0, selected_server['health'] - random.randint(1, 5))
        selected_server['response_time'] += random.randint(-5, 5)
        selected_server['failure_rate'] = min(0.2, selected_server['failure_rate'] + random.uniform(-0.01, 0.01))

        reward = self.reward_function(selected_server)
        self.time_step += 1

        self.time_steps_history.append(self.time_step)
        self.demand_history.append(self.demand)
        self.storage_history.append(self.storage)
        self.rewards_history.append(reward)

        return self.get_observation(), reward, False, {}

    def action_space(self):
        return np.arange(self.num_servers)
