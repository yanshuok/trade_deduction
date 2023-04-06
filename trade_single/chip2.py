from envs import Env
from PPO import ProximalPolicyOptimization, Actor, Critic, ReplayBuffer


ts_per_batch = 64
env = Env(server_addr='http://192.168.3.46:30001/')
usa_actor = Actor(env.state_dim, len(env.actions['usa']))
usa_critic = Critic(env.state_dim)
agent_usa = ProximalPolicyOptimization(actor=usa_actor, critic=usa_critic, clip_rate=0.2, replay_buffer=ReplayBuffer(size=ts_per_batch // 2, state_dim=env.state_dim))
china_actor = Actor(env.state_dim, len(env.actions['chn']))
china_critic = Critic(env.state_dim)
agent_chn = ProximalPolicyOptimization(actor=china_actor, critic=china_critic, clip_rate=0.2, replay_buffer=ReplayBuffer(size=ts_per_batch // 2, state_dim=env.state_dim))
agent_chn.load("china")
agent_usa.load("usa")
episode = 0
state = env.reset()
eps_reward = {'chn': 0, 'usa': 0}
eps_values = {'chn': 0, 'usa': 0}
for i in range(10000):
    if episode % 100 == 0:
        agent_usa.save("usa")
        agent_chn.save("china")
        env.save()
    for j in range(ts_per_batch):
        if env.ts % 2 == 0:
            region = 'usa'
            agent = agent_usa
        else:
            region = 'chn'
            agent = agent_chn
        available_actions = env.get_available_actions(region)
        # print(available_actions)
        a, pk = agent.select_action(state, available_actions)
        value = agent.get_value(state)
        state_, reward, done = env.step(a, region)
        agent.replay_buffer.save(state, a, reward, value, pk)
        state = state_
        eps_reward[region] += reward
        eps_values[region] = agent.get_value(state_)
        if done or j == ts_per_batch - 1:
            if done:
                print(episode, eps_reward['chn'], eps_reward['usa'])
                episode += 1
                state = env.reset()
            agent_chn.replay_buffer.final_path(eps_values['chn'])
            agent_usa.replay_buffer.final_path(eps_values['usa'])
            eps_reward = {'chn': 0, 'usa': 0}

    agent_chn.train()
    agent_usa.train()