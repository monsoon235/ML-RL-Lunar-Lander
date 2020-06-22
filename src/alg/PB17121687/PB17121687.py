from gym.spaces import Discrete

from src.alg.RL_alg import RL_alg
from src.utils.misc_utils import get_params_from_file

from src.alg.PB17121687.lunar_lander.agent import LunarLanderAgent


class PB17121687(RL_alg):
    agent: LunarLanderAgent

    def __init__(self, ob_space, ac_space):
        super().__init__()
        assert isinstance(ac_space, Discrete)

        self.team = ['PB17121687']  # 记录队员学号
        self.config = get_params_from_file('src.alg.PB17121687.rl_configs', params_name='params')  # 传入参数

        self.ac_space = ac_space
        self.state_dim = ob_space.shape[0]
        self.action_dim = ac_space.n

        self.agent = LunarLanderAgent()
        self.agent.load('src/alg/PB17121687/lunar_lander/saved_model')

    def step(self, state):
        return self.agent.act(state)

    def explore(self, obs):
        raise NotImplementedError

    def test(self):
        print('ok1')
