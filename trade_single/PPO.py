import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from scipy.signal import lfilter
from tqdm import tqdm


def discount_cumsum(x, discount):
    advantages = lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    return advantages


class ReplayBuffer():

    def __init__(self, size=5000, state_dim=4, gamma=0.99):
        self.gamma = gamma
        if type(state_dim) == int:
            self.state_buffer = np.zeros([size, state_dim])
        elif type(state_dim) == list:
            self.state_buffer = np.zeros([size] + [state_dim])
        elif type(state_dim) == tuple:
            self.state_buffer = np.zeros([size] + [x for x in state_dim])
        self.action_buffer = np.zeros([size])
        self.reward_buffer = np.zeros([size])
        self.value_buffer = np.zeros([size])
        self.old_logp_buffer = np.zeros([size])
        self.return_buffer = np.zeros([size])
        self.adv_buffer = np.zeros([size])
        self.start_point, self.end_point = 0, 0

    def save(self, state, action, reward, value, old_logp):
        self.state_buffer[self.end_point] = state
        self.action_buffer[self.end_point] = action
        self.reward_buffer[self.end_point] = reward
        self.value_buffer[self.end_point] = value
        self.old_logp_buffer[self.end_point] = old_logp
        self.end_point += 1

    def final_path(self, last_value):
        rewards = self.reward_buffer[self.start_point: self.end_point]
        values = np.append(self.value_buffer[self.start_point: self.end_point], last_value)
        deltas = rewards + self.gamma * values[1:] - values[:-1]
        self.adv_buffer[self.start_point: self.end_point] = discount_cumsum(deltas, self.gamma)
        self.return_buffer[self.start_point: self.end_point] = discount_cumsum(rewards, self.gamma)
        self.start_point = self.end_point

    def get(self):
        mean, std = np.mean(self.adv_buffer[:self.end_point]), np.std(self.adv_buffer[:self.end_point])
        advs = (self.adv_buffer[:self.end_point] - mean) / std
        states, actions, ret, logp = self.state_buffer[:self.end_point], self.action_buffer[:self.end_point], self.return_buffer[:self.end_point], self.old_logp_buffer[:self.end_point]
        self.start_point, self.end_point = 0, 0
        return states, actions, ret, advs, logp


class Actor(nn.Module):

    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(state_dim, 10)
        self.linear2 = nn.Linear(10, action_dim)

    def forward(self, inputs):
        a1 = F.relu(self.linear1(inputs))
        p = F.log_softmax(self.linear2(a1), dim=1)
        return p


class Critic(nn.Module):
    
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim, 100)
        self.linear2 = nn.Linear(100, 1)

    def forward(self, inputs):
        a1 = F.relu(self.linear1(inputs))
        a2 = self.linear2(a1)
        return a2


class ProximalPolicyOptimization(object):

    def __init__(self, actor:nn.Module, critic:nn.Module, replay_buffer: ReplayBuffer=None, clip_rate:float=0.2, batch_size:int=64, device=None):
        self.actor = actor.to(device)
        self.critic = critic.to(device)
        self.device = device
        self.replay_buffer = replay_buffer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), 0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), 0.01)
        self.clip_rate = clip_rate
        self.batch_size = batch_size

    def select_action(self, state, available_actions=None):
        p = self.actor(torch.FloatTensor(state).to(self.device))
        if available_actions is not None:
            p1 = torch.multiply(torch.exp(p), torch.FloatTensor(np.reshape(available_actions, (1, -1))))
            if torch.sum(p1).detach().numpy() == 0:
                a = torch.squeeze(torch.multinomial(torch.multiply(torch.FloatTensor([0.5, 0.5, 0.5, 0.5]), torch.FloatTensor(np.reshape(available_actions, (1, -1)))), 1), 0)
            else:
                p1 = p1 / torch.sum(p1)
                a = torch.squeeze(torch.multinomial(p1, 1), 0)
        else:
            a = torch.squeeze(torch.multinomial(torch.exp(p), 1), 0)
        log_pi = torch.take(p, a)
        return a.cpu().numpy()[0], log_pi.detach().cpu().numpy()[0]

    def greedy_action(self, state, available_actions=None):
        p = self.actor(torch.FloatTensor(state))
        if available_actions is not None:
            p1 = torch.multiply(torch.exp(p), torch.FloatTensor(np.reshape(available_actions, (1, -1))))
            a = torch.squeeze(torch.argmax(p1, 1), 0)
        else:
            a = torch.squeeze(torch.argmax(p, 1), 0)
        log_pi = torch.take(p, a)
        return a.numpy(), log_pi.detach().numpy()

    def get_value(self, state):
        return torch.squeeze(self.critic(torch.FloatTensor(state).to(self.device)), 0).detach().cpu().numpy()[0]

    def train(self):
        states, actions, rets, advs, logps = self.replay_buffer.get()
        states, actions, rets, advs, logps = torch.FloatTensor(states), torch.LongTensor(actions), torch.FloatTensor(rets), torch.FloatTensor(advs), torch.FloatTensor(logps)

        dataset = torch.utils.data.TensorDataset(states, actions, rets, advs, logps)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
        for i in range(10):
            for s, a, r, adv, logp in dataloader:
                s, a, r, adv, logp = s.to(self.device), a.to(self.device), r.to(self.device), adv.to(self.device), logp.to(self.device)
                self.actor_optimizer.zero_grad()
                p = self.actor(s)
                ratio = torch.exp(torch.squeeze(torch.take_along_dim(p, torch.unsqueeze(a, -1), 1), 1) - logp)
                actor_loss = -torch.mean(torch.minimum(torch.clip(ratio, 1 + self.clip_rate, 1 - self.clip_rate) * adv, ratio * adv))
                actor_loss.backward()
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                value = self.critic(s)
                critic_loss = torch.mean(torch.square(value - r))
                critic_loss.backward()
                self.critic_optimizer.step()

    def save(self, path):
        torch.save(self.actor, path + "_actor.pth")
        torch.save(self.critic, path + "_critic.pth")

    def load(self, path):
        self.actor = torch.load(path + "_actor.pth")
        self.critic = torch.load(path + "_critic.pth")


if __name__ == '__main__':
    max_frame_per_epoch = 2000
    env = gym.make("CartPole-v1")
    replay_buffer = ReplayBuffer(size=max_frame_per_epoch, state_dim=4)
    actor = Actor(state_dim=4, action_dim=2)
    critic = Critic(state_dim=4)
    ppo = ProximalPolicyOptimization(actor=actor, critic=critic, replay_buffer=replay_buffer, clip_rate=0.2)

    state, _ = env.reset()
    state = state[np.newaxis, :]
    episode = 0
    for i in range(10000):
        eps_reward = 0
        for j in range(max_frame_per_epoch):
            a, pk = ppo.select_action(state)
            value = ppo.get_value(state)
            state_, reward, terminated, truncation, _ = env.step(action=a)
            replay_buffer.save(state, a, reward, value, pk)
            state = state_[np.newaxis, :]
            eps_reward += reward
            if j == max_frame_per_epoch - 1 or terminated or truncation:
                last_value = ppo.get_value(state)
                if terminated or truncation:
                    print(episode, eps_reward)
                    eps_reward = 0
                    episode += 1
                    state, _ = env.reset()
                    state = state[np.newaxis, :]
                replay_buffer.final_path(last_value)
        ppo.train()