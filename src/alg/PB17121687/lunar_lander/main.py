from src.alg.PB17121687.lunar_lander.config import config
from src.alg.PB17121687.lunar_lander.agent import LunarLanderAgent
import gym

floatX = config['floatX']
device = config['device']


def train():
    env = gym.make('LunarLander-v2')
    agent = LunarLanderAgent()
    agent.train(
        env=env,
        learning_rate=5e-4,
        max_episode=2000,
        max_t=1000,
        reply_memory_size=-1,
        mini_batch_size=64,
        eps_start=1.0,
        eps_end=0.01,
        eps_decay=0.995,
        tau=1e-3,
        gamma=0.99
    )
    agent.save('src/alg/PB17121687/lunar_lander/saved_model')


def show():
    env = gym.make('LunarLander-v2')
    agent = LunarLanderAgent()
    agent.load('src/alg/PB17121687/lunar_lander/saved_model')
    for j in range(1000):
        obs = env.reset()
        done = False
        ep_ret = 0
        ep_len = 0
        while not done:
            env.render()
            # Take deterministic actions at test time
            ac = agent.act(obs)
            obs, reward, done, ss = env.step(ac)
            print(obs)
            print(reward)
            ep_ret += reward
            ep_len += 1
        print("TestEpRet", ep_ret)
        print("TestEpLen", ep_len)


if __name__ == '__main__':
    train()
    show()
