import random
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class AllOnes(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, state_size=4, action_size=2):
        self.state_size = state_size
        self.action_size = action_size
        state = [random.random() for _ in range(self.state_size)]

    def step(self, action):
        pass

    def reset(self):
        state = [random.random() for _ in range(self.state_size)]

    def render(self, mode='human'):
        pass

    def close(self):
        pass
