import numpy as np


class Random_Agent:
    def __init__(self, env):
        self.state_num = env.state_space_size
        self.action_num = env.action_space_size


    def take_action(self, state):
        return np.random.randint(0, self.action_num)
