import time

from gym.spaces import Discrete

from src.alg.RL_alg import RL_alg
from src.utils.misc_utils import get_params_from_file

import numpy as np
from PIL import Image


class PB00000000(RL_alg):
    def __init__(self, ob_space, ac_space):
        super().__init__()
        assert isinstance(ac_space, Discrete)

        self.team = ['PB00000000', 'PB10000000', 'PB20000000']  # 记录队员学号
        self.config = get_params_from_file('src.alg.PB00000000.rl_configs', params_name='params')  # 传入参数

        self.ac_space = ac_space
        self.state_dim = ob_space.shape[0]
        self.action_dim = ac_space.n

    def step(self, state: np.ndarray):
        action = self.ac_space.sample()
        # Image.fromarray(state[2:201, 8:, :]).save('C:\\Users\\monsoon\\Desktop\\state.jpg')
        # exit()
        return action

    def explore(self, obs):
        raise NotImplementedError

    def test(self):
        print('ok1')
