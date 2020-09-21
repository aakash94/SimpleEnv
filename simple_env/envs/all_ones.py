import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class AllOnes(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, state_size=4, action_size=2, timestep_limit=1024):
        self.state_size = state_size
        self.action_size = action_size
        self.timestep_limit = timestep_limit

        self.reference = np.ones(self.action_size)

        self.max_action = np.inf
        self.min_action = -np.inf

        self.observation_space = spaces.Box(low=-0,
                                            high=1,
                                            shape=(self.state_size,),
                                            dtype=np.float32)

        self.action_space = spaces.Box(low=self.min_action,
                                       high=self.max_action,
                                       shape=(self.action_size,),
                                       dtype=np.float32)

        self.reset()

    def step(self, action):

        self.step_count += 1

        if self.step_count >= self.timestep_limit:
            self.done = True

        self.state = [np.random.rand(self.state_size)]

        mse = ((self.reference - action) ** 2).mean()
        reward = -mse

        return self.state, reward, self.done, {}

    def reset(self):
        self.step_count = 0
        self.state = [np.random.rand(self.state_size)]
        self.done = False
        return self.state

    def render(self, mode='human'):
        print(self.state)

    def close(self):
        self.step_count = 0
        self.done = False

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


if __name__ == '__main__':

    eao = AllOnes()
    s = eao.reset()
    eao.render()
    a = np.array([1.0,1.0])
    s_, r, d, _ = eao.step(a)
    print(r)
    eao.render()
