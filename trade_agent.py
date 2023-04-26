from envs import Env
from PPO import ProximalPolicyOptimization, Actor, Critic, ReplayBuffer
import numpy as np

ts_per_batch = 480
env = Env(server_addr='http://192.168.3.46:30001/')
usa_actor = Actor(env.state_dim, env.action_dim['usa'])
usa_critic = Critic(env.state_dim)
agent_usa = ProximalPolicyOptimization(actor=usa_actor, critic=usa_critic, clip_rate=0.2, replay_buffer=ReplayBuffer(size=ts_per_batch // 2, state_dim=env.state_dim, action_dim=env.action_dim['usa']))
china_actor = Actor(env.state_dim, env.action_dim['chn'])
china_critic = Critic(env.state_dim)
agent_chn = ProximalPolicyOptimization(actor=china_actor, critic=china_critic, clip_rate=0.2, replay_buffer=ReplayBuffer(size=ts_per_batch // 2, state_dim=env.state_dim, action_dim=env.action_dim['chn']))
agent_chn.load("china")
agent_usa.load("usa")
episode = 0
state = env.reset()
eps_reward = {'chn': 0, 'usa': 0}
eps_values = {'chn': 0, 'usa': 0}
error_count = 0
for i in range(10000):
    for j in range(ts_per_batch):
        if env.ts % 2 == 0:
            region = 'usa'
            rival_region = 'chn'
            agent = agent_usa
        else:
            region = 'chn'
            rival_region = 'usa'
            agent = agent_chn
        # try:
        a, pk, value = agent.select_action_and_value(state, round=env.round)
        state_, reward, done = env.step(a, region)
        agent.replay_buffer.save(state, a, reward[region] - reward[rival_region], value, pk)
        state = state_
        eps_reward[region] += reward[region] - reward[rival_region]
        eps_values[region] = agent.get_value(state_)
        # except KeyError as e:
        #     print("error")
        #     error_count += 1
        #     done = True
        if done or j == ts_per_batch - 1:
            if done:
                print(episode, eps_reward['chn'], eps_reward['usa'], error_count)
                episode += 1
                state = env.reset()
                if episode % 5 == 0:
                    print("saving!")
                    agent_usa.save("usa")
                    agent_chn.save("china")
                    env.save()
            agent_chn.replay_buffer.final_path(eps_values['chn'])
            agent_usa.replay_buffer.final_path(eps_values['usa'])
            eps_reward = {'chn': 0, 'usa': 0}
    print("training")
    agent_chn.train()
    agent_usa.train()