import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding


class CopyCat(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, state_size=4, action_size=2, timestep_limit=1024):
        self.state_size = state_size
        self.action_size = action_size
        self.timestep_limit = timestep_limit

        self.max_action = np.inf
        self.min_action = -np.inf

        self.state_size_ref_one = np.ones(self.state_size)
        self.state_size_ref_zero = np.zeros(self.state_size)

        self.action_ref_one = np.ones(self.action_size)
        self.action_ref_zero = np.zeros(self.action_size)

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

        reference = self.action_ref_one
        if self.state[0] == 0:
            reference = self.action_ref_zero

        mse = ((reference - action) ** 2).mean()
        reward = -mse

        if 0.5 > np.random.rand():
            self.state = self.state_size_ref_one
        else:
            self.state = self.state_size_ref_zero

        return self.state, reward, self.done, {}

    def reset(self):
        self.step_count = 0
        if 0.5 > np.random.rand():
            self.state = self.state_size_ref_one
        else:
            self.state = self.state_size_ref_zero

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
    cc = CopyCat()
    s = cc.reset()

    cc.render()
    a = np.array([1.0, 1.0])
    print("action ", a)
    s_, r, d, _ = cc.step(a)
    print("reward ", r)

    cc.render()
    a = np.array([0.0, 0.0])
    print("action ", a)
    s_, r, d, _ = cc.step(a)
    print("reward ", r)
