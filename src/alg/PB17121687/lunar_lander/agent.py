from typing import Union, List, Tuple

import torch
from gym.spaces import Box

from src.alg.PB17121687.lunar_lander.config import config
from src.alg.PB17121687.lunar_lander.model import DuelingDoubleDQN
import math
import random
import gym

import numpy as np

floatX = config['floatX']
device = config['device']


class ReplyMemory:
    size: Union[int, float]
    memory: List[dict]

    def __init__(self, size: Union[int, float]) -> None:
        self.memory = []
        if size >= 1:
            self.size = size
        else:
            self.size = math.inf

    def add(self, record: dict) -> bool:
        if len(self.memory) + 1 <= self.size:
            self.memory.append(record)
            return True
        else:
            return False

    def sample(self, batch_size: int) -> Tuple[int, List[dict]]:
        if len(self.memory) <= batch_size:
            return len(self.memory), self.memory
        else:
            selection = []
            while len(selection) < batch_size:
                i = random.randint(0, len(self.memory) - 1)
                if i not in selection:
                    selection.append(i)
            return batch_size, [self.memory[i] for i in selection]


class LunarLanderAgent:
    env: gym.Wrapper
    memory: ReplyMemory
    Q_local: DuelingDoubleDQN
    Q_target: DuelingDoubleDQN

    def __init__(self) -> None:
        self.Q_local = DuelingDoubleDQN()
        self.Q_target = DuelingDoubleDQN()

    def update_target(self, tau: float):
        self.Q_target.fc1.weight = tau * self.Q_local.fc1.weight + (1 - tau) * self.Q_target.fc1.weight
        self.Q_target.fc2.weight = tau * self.Q_local.fc2.weight + (1 - tau) * self.Q_target.fc2.weight
        self.Q_target.fc_v1.weight = tau * self.Q_local.fc_v1.weight + (1 - tau) * self.Q_target.fc_v1.weight
        self.Q_target.fc_v2.weight = tau * self.Q_local.fc_v2.weight + (1 - tau) * self.Q_target.fc_v2.weight
        self.Q_target.fc_a1.weight = tau * self.Q_local.fc_a1.weight + (1 - tau) * self.Q_target.fc_a1.weight
        self.Q_target.fc_a2.weight = tau * self.Q_local.fc_a2.weight + (1 - tau) * self.Q_target.fc_a2.weight
        self.Q_target.fc1.bias = tau * self.Q_local.fc1.bias + (1 - tau) * self.Q_target.fc1.bias
        self.Q_target.fc2.bias = tau * self.Q_local.fc2.bias + (1 - tau) * self.Q_target.fc2.bias
        self.Q_target.fc_v1.bias = tau * self.Q_local.fc_v1.bias + (1 - tau) * self.Q_target.fc_v1.bias
        self.Q_target.fc_v2.bias = tau * self.Q_local.fc_v2.bias + (1 - tau) * self.Q_target.fc_v2.bias
        self.Q_target.fc_a1.bias = tau * self.Q_local.fc_a1.bias + (1 - tau) * self.Q_target.fc_a1.bias
        self.Q_target.fc_a2.bias = tau * self.Q_local.fc_a2.bias + (1 - tau) * self.Q_target.fc_a2.bias

    def env_step(self, action: int, render=False) -> Tuple[torch.Tensor, float, bool]:
        obs, reward, done, _ = self.env.step(action)
        if render:
            self.env.render()
        if done:
            self.env.reset()
        return torch.tensor(obs, dtype=floatX), reward, done

    def selection_action(self, state: torch.Tensor, epsilon: float = 0) -> int:
        if random.random() <= epsilon:
            return self.env.action_space.sample()
        else:
            in_date = state.reshape((1,) + state.shape).to(device)
            q_values = self.Q_local.forward(in_date)
            assert q_values.shape == (1, 4)
            return int(q_values.reshape((4,)).argmax())

    def learn(self,
              state: torch.Tensor,
              learning_rate: float,
              mini_batch_size: int,
              tau: float,
              gamma: float
              ):

        bs, batch = self.memory.sample(mini_batch_size)
        states = torch.empty(size=(bs,) + state.shape, dtype=floatX, device=device)
        actions = torch.empty(size=(bs,), dtype=torch.int64)
        rewards = torch.empty(size=(bs,), dtype=floatX)
        states_next = torch.empty(size=(bs,) + state.shape, dtype=floatX, device=device)
        dones = torch.empty(size=(bs,), dtype=torch.int)

        for i in range(bs):
            states[i] = batch[i]['state']
            actions[i] = batch[i]['action']
            rewards[i] = batch[i]['reward']
            states_next[i] = batch[i]['state_next']
            dones[i] = batch[i]['done']

        # Double DQN
        q_arg_max = self.Q_local.forward(states_next).cpu()
        a_prime = q_arg_max.argmax(dim=1)
        q_target_next = self.Q_target.forward(states_next).cpu().gather(dim=1, index=a_prime.reshape((bs, 1))).flatten()

        # DQN
        # q_target_next = self.Q_target.forward(states_next).cpu().max(dim=1)

        ys = rewards + gamma * (1 - dones) * q_target_next
        qs = self.Q_local.forward(states).cpu().gather(dim=1, index=actions.reshape((bs, 1))).flatten()

        loss = (ys - qs) ** 2
        eta = torch.zeros(size=(bs, 4), dtype=floatX, device=device)
        for i in range(bs):
            eta[i, actions[i]] = -2 * (ys[i] - qs[i])

        self.Q_local.backward(eta, learning_rate)
        self.update_target(tau)
        return loss

        # for episode in range(max_episode):
        #     a = self.selection_action(s, epsilon)
        #     s_next, r, done_next = self.env_step(a, episode % 20 == 0)
        #     self.memory.add({
        #         's': s, 'a': a, 'r': r,
        #         's_next': s_next, 'done_next': done_next
        #     })
        #
        #     bs, batch = self.memory.sample(mini_batch_size)
        #     r_all = torch.tensor([record['r'] for record in batch], dtype=floatX)
        #     done_all = torch.tensor([record['done_next'] for record in batch], dtype=floatX)
        #     in_data = torch.empty(size=(bs,) + s.shape, dtype=floatX, device=device)
        #     for i in range(bs):
        #         in_data[i] = batch[i]['s_next']
        #     q_all = self.Q_target.forward(in_data).cpu()
        #     y = r_all + gamma * (1 - done_all) * q_all.max(dim=1)[0]
        #     assert y.shape == (bs,)
        #
        #     in_data = torch.empty(size=(bs,) + s.shape, dtype=floatX, device=device)
        #     for i in range(bs):
        #         in_data[i] = batch[i]['s']
        #     q_all = self.Q_local.forward(in_data).cpu()
        #     assert q_all.shape == (bs, 4)
        #     q = torch.tensor([q_all[i, batch[i]['a']] for i in range(bs)], dtype=floatX)
        #     assert q.shape == (bs,)
        #
        #     loss = (y - q) ** 2
        #     print(loss.sum())
        #
        #     eta = torch.zeros(size=(bs, 4), dtype=floatX, device=device)
        #     for i in range(bs):
        #         eta[i, batch[i]['a']] = -2 * (y[i] - q[i])
        #
        #     self.Q_local.backward(eta, learning_rate)
        #     self.update_target(tau)
        #     s = s_next

    def train(self,
              env: gym.Wrapper,
              learning_rate: float,
              max_episode: int,
              max_t: int,
              reply_memory_size: int,
              mini_batch_size: int,
              eps_start: float,
              eps_end: float,
              eps_decay: float,
              tau: float,
              gamma: float
              ) -> List[float]:
        # 初始化环境
        self.env = env
        self.memory = ReplyMemory(reply_memory_size)
        eps = eps_start  # 随着训练次数的上升， eps 逐渐下降
        score_list = []
        for episode in range(max_episode):
            # 初始状态
            print('episode =', episode)
            obs = self.env.reset()
            state = torch.tensor(obs, dtype=floatX)
            score = 0
            for t in range(max_t):
                action = self.selection_action(state, eps)
                state_next, reward, done = self.env_step(action, t % 20 == 0)
                self.memory.add({
                    'state': state,
                    'action': action,
                    'reward': reward,
                    'state_next': state_next,
                    'done': done
                })
                loss = self.learn(state, learning_rate=learning_rate,
                                  mini_batch_size=mini_batch_size,
                                  tau=tau, gamma=gamma)
                state = state_next
                score += reward
                if done:
                    break
            score_list.append(score)  # 记录分数
            print('score =', score)
            eps = max(eps_end, eps_decay * eps)
            if (episode + 1) % 10 == 0:
                print('avg score =', np.mean(score_list[episode - 9:episode + 1]))
        return score_list

    def save(self, path: str):
        self.Q_target.save(path)

    def load(self, path: str):
        self.Q_target.load(path)

    def act(self, obs: Box):
        state = torch.tensor(obs, dtype=floatX, device=device)
        state = state.reshape((1,) + state.shape)
        q_values = self.Q_target.forward(state)
        assert q_values.shape == (1, 4)
        return int(q_values.reshape((4,)).argmax())
