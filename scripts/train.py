from time import sleep
from typing import Tuple, Set, List

from cnn.cnn import DuelingDQN
import torch
from cnn.config import config
import configs.main_setting
import gym
import random

floatX = config['floatX']
device = config['device']

env = gym.make(configs.main_setting.params['env_name'])
obs = env.reset()


def evaluate(env, policy, num_evaluate_episodes, is_render):
    for j in range(num_evaluate_episodes):
        obs = env.reset()
        done = False
        ep_ret = 0
        ep_len = 0
        while not done:
            if is_render:
                env.render()
            # Take deterministic actions at test time
            ac = policy.step(obs)
            obs, reward, done, ss = env.step(ac)
            print(reward, done, ss)
            ep_ret += reward
            ep_len += 1
        policy.logkv_mean("TestEpRet", ep_ret)
        policy.logkv_mean("TestEpLen", ep_len)
    policy.dumpkvs()


# 选择 action
def select_action(epsilon: float, Q: DuelingDQN, s: torch.Tensor) -> int:
    if random.random() <= epsilon:
        return env.action_space.sample()
    else:
        Q_values = Q.forward(s.reshape((1,) + s.shape).cuda(device=device))
        assert Q_values.shape == (1, 18)
        Q_values = Q_values.reshape((18,))
        return int(Q_values.argmax())


def execute_action(action: int) -> Tuple[torch.Tensor, float, bool]:
    obs, reward, done, _ = env.step(action)
    env.render()
    if done:
        env.reset()
    obs = torch.tensor(obs[2:201, 8:, :], dtype=floatX) / 255
    return obs, reward, done


def sample_transitions(reply_memory: List[dict], batch_size: int) -> List[dict]:
    size = len(reply_memory)
    if size < batch_size:
        return reply_memory
    else:
        selection = []
        while len(selection) < batch_size:
            index = random.randint(0, size - 1)
            if index not in selection:
                selection.append(index)
        return [reply_memory[i] for i in selection]


def minibatch_GD():
    pass


def update_Q_target(Q: DuelingDQN, Q_target: DuelingDQN, tau: float):
    Q_target.conv1.weight = tau * Q.conv1.weight + (1 - tau) * Q_target.conv1.weight
    Q_target.conv2.weight = tau * Q.conv2.weight + (1 - tau) * Q_target.conv2.weight
    Q_target.conv3.weight = tau * Q.conv3.weight + (1 - tau) * Q_target.conv3.weight
    Q_target.conv1.bias = tau * Q.conv1.bias + (1 - tau) * Q_target.conv1.bias
    Q_target.conv2.bias = tau * Q.conv2.bias + (1 - tau) * Q_target.conv2.bias
    Q_target.conv3.bias = tau * Q.conv3.bias + (1 - tau) * Q_target.conv3.bias
    Q_target.fc_v1.weight = tau * Q.fc_v1.weight + (1 - tau) * Q_target.fc_v1.weight
    Q_target.fc_v2.weight = tau * Q.fc_v2.weight + (1 - tau) * Q_target.fc_v2.weight
    Q_target.fc_a1.weight = tau * Q.fc_a1.weight + (1 - tau) * Q_target.fc_a1.weight
    Q_target.fc_a1.weight = tau * Q.fc_a1.weight + (1 - tau) * Q_target.fc_a1.weight
    Q_target.fc_v1.bias = tau * Q.fc_v1.bias + (1 - tau) * Q_target.fc_v1.bias
    Q_target.fc_v2.bias = tau * Q.fc_v2.bias + (1 - tau) * Q_target.fc_v2.bias
    Q_target.fc_a1.bias = tau * Q.fc_a1.bias + (1 - tau) * Q_target.fc_a1.bias
    Q_target.fc_a1.bias = tau * Q.fc_a1.bias + (1 - tau) * Q_target.fc_a1.bias


if __name__ == '__main__':
    reply_memory = []
    Q = DuelingDQN()
    Q_target = DuelingDQN()
    sleep(5)
    M = 100
    T = 100
    epsilon = 0.1
    tau = 0.99
    gamma = 0.9
    target_bs = 2

    obs = env.reset()
    s = torch.tensor(obs[2:201, 8:, :], dtype=floatX) / 255
    for episode in range(M):
        for t in range(T):
            a = select_action(epsilon, Q, s)  # 选择 at
            s_next, r, done_next = execute_action(a)  # 执行 at

            # 更新 reply_memory
            reply_memory.append({
                's': s,
                'a': a,
                'r': r,
                's_next': s_next,
                'done_next': done_next
            })
            minibatch = sample_transitions(reply_memory, target_bs)
            bs = len(minibatch)

            # 计算 y
            y = torch.empty(size=(bs,), dtype=floatX)
            for i in range(bs):
                sample = minibatch[i]
                if sample['done_next']:
                    y[i] = sample['r']
                else:
                    si_next = torch.tensor(sample['s_next'], dtype=floatX)
                    si_next = si_next.reshape((1,) + si_next.shape)
                    y[i] = sample['r'] + gamma * float(Q_target.forward(si_next.cuda(device=device)).max())

            s_all = torch.empty(size=(bs, 199, 152, 3), dtype=floatX)
            for i in range(bs):
                s_all[i] = minibatch[i]['s']
            q_all = Q.forward(s_all.cuda(device=device)).cpu()
            assert q_all.shape == (bs, 18)

            q = torch.tensor([q_all[i][minibatch[i]['a']] for i in range(bs)], dtype=floatX)
            assert q.shape == (bs,)

            loss = (y - q) ** 2
            print(q)
            print(loss)

            eta = torch.zeros(size=(bs, 18), dtype=floatX)
            for i in range(bs):
                eta[i, minibatch[i]['a']] = -2 * (y[i] - q[i])

            Q.backward(eta.cuda(device=device))
            update_Q_target(Q, Q_target, tau)
