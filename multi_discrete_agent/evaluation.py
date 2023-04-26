from envs import Env
from PPO import ProximalPolicyOptimization, Actor, Critic


env = Env(server_addr='http://192.168.3.46:30001/')
usa_actor = Actor(env.state_dim, env.action_dim['usa'])
usa_critic = Critic(env.state_dim)
agent_usa = ProximalPolicyOptimization(actor=usa_actor, critic=usa_critic, clip_rate=0.2, replay_buffer=None)
china_actor = Actor(env.state_dim, env.action_dim['chn'])
china_critic = Critic(env.state_dim)
agent_chn = ProximalPolicyOptimization(actor=china_actor, critic=china_critic, clip_rate=0.2, replay_buffer=None)
agent_chn.load("china")
agent_usa.load("usa")
state = env.reset()
eps_reward = {'chn': 0, 'usa': 0}
action_seq = [[10, 0, 0, 0, 0, 0], [10, 0, 0, 0, 0, 0],
              [0, 2, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0],
              [0, 0, 0, 18, 0, 0], [0, 0, 0, 12, 0, 0],
              [0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 1, 0],
              [0, 0, 0, 0, 0, 6], [0, 0, 0, 0, 0, 6]]
while True:
    if env.ts % 2 == 0:
        region = 'usa'
        agent = agent_usa
    else:
        region = 'chn'
        agent = agent_chn
    # available_actions = env.get_available_actions(region)
    a = agent.greedy_action(state, round=env.round)
    value = agent.get_value(state)
    state_, reward, done = env.step(action_seq[env.ts], region)
    print(reward)
    # agent.replay_buffer.save(state, a, reward, value, pk)
    state = state_
    eps_reward[region] += reward[region]
    if done:
        break
print(eps_reward['chn'], eps_reward['usa'])