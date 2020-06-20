from time import sleep


def evaluate(env, policy, num_evaluate_episodes, is_render):
    for j in range(num_evaluate_episodes):
        obs = env.reset()
        done = False
        ep_ret = 0
        ep_len = 0
        pre_ale = 4
        while not done:
            if is_render:
                env.render()
            # Take deterministic actions at test time 
            ac = policy.step(obs)
            obs, reward, done, ss = env.step(ac)
            print(reward, done, ss)
            ale = ss['ale.lives']
            if ale != pre_ale:
                pre_ale = ale
                sleep(10)
            ep_ret += reward
            ep_len += 1
        policy.logkv_mean("TestEpRet", ep_ret)
        policy.logkv_mean("TestEpLen", ep_len)
    policy.dumpkvs()
