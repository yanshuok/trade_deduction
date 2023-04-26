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

    def __init__(self, size=5000, state_dim=4, action_dim=2, gamma=0.99):
        self.gamma = gamma
        if type(state_dim) == int:
            self.state_buffer = np.zeros([size, state_dim])
        elif type(state_dim) == list:
            self.state_buffer = np.zeros([size] + [state_dim])
        elif type(state_dim) == tuple:
            self.state_buffer = np.zeros([size] + [x for x in state_dim])
        if type(action_dim) == int:
            self.action_buffer = np.zeros([size])
        elif type(action_dim) == list:
            self.action_buffer = np.zeros([size, len(action_dim)])
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
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.linear1 = nn.Linear(state_dim, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, sum(action_dim))

    def forward(self, inputs):
        a1 = F.relu(self.linear1(inputs))
        a2 = F.relu(self.linear2(a1))
        a3 = self.linear3(a2)
        return a3


class Critic(nn.Module):

    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_dim, 1000)
        self.linear2 = nn.Linear(1000, 1000)
        self.linear3 = nn.Linear(1000, 1)

    def forward(self, inputs):
        a1 = F.relu(self.linear1(inputs))
        a2 = F.relu(self.linear2(a1))
        a3 = self.linear3(a2)
        return a3


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

    def get_action_and_value(self, state, action=None, round=0):
        logits = self.actor(torch.FloatTensor(state).to(self.device))
        split_logits = torch.split(logits, self.actor.action_dim, dim=1)
        multi_categoricals = [torch.distributions.categorical.Categorical(logits=x) for x in split_logits]
        # print([x.probs for x in multi_categoricals])
        value = self.critic(torch.FloatTensor(state).to(self.device))
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
            for i in range(len(self.actor.action_dim)):
                if i != round:
                    action[i, 0] = 0
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action.T, logprob.sum(0), value, entropy.sum(0)

    def select_action_and_value(self, state, round=0):
        action, logprob, value, _ = self.get_action_and_value(state, round=round)
        return np.squeeze(action.detach().cpu().numpy(), 0), logprob.detach().cpu().numpy()[0], value.detach().cpu().numpy()[0, 0]

    def greedy_action(self, state, round=0):
        logits = self.actor(torch.FloatTensor(state).to(self.device))
        split_logits = torch.split(logits, self.actor.action_dim, dim=1)
        # print([F.softmax(x, 1) for x in split_logits])
        bak_action = [torch.argmax(x, 1) for x in split_logits]
        action = [0 for _ in range(len(self.actor.action_dim))]
        action[round] = bak_action[round].detach().cpu().numpy()[0]
        return action

    def get_value(self, state):
        return torch.squeeze(self.critic(torch.FloatTensor(state).to(self.device)), 0).detach().cpu().numpy()[0]

    def train(self):
        states, actions, rets, advs, logps = self.replay_buffer.get()
        states, actions, rets, advs, logps = torch.FloatTensor(states), torch.LongTensor(actions), torch.FloatTensor(rets), torch.FloatTensor(advs), torch.FloatTensor(logps)

        dataset = torch.utils.data.TensorDataset(states, actions, rets, advs, logps)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64)
        # for i in range(5):
        for s, a, r, adv, logp in dataloader:
            a, r, adv, logp = a.to(self.device), r.to(self.device), adv.to(
                self.device), logp.to(self.device)
            self.actor_optimizer.zero_grad()
            _, new_logporb, new_value, entropy = self.get_action_and_value(s, a.T)
            ratio = torch.exp(new_logporb - logp)
            actor_loss = -torch.mean(
                torch.minimum(torch.clip(ratio, 1 + self.clip_rate, 1 - self.clip_rate) * adv, ratio * adv)) + entropy.mean()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss = torch.mean(torch.square(new_value - r[:, None]))
            critic_loss.backward()
            self.critic_optimizer.step()

    def save(self, path):
        torch.save(self.actor, path + "_actor.pth")
        torch.save(self.critic, path + "_critic.pth")

    def load(self, path):
        if os.path.exists(path + "_actor.pth"):
            self.actor = torch.load(path + "_actor.pth")
        if os.path.exists(path + "_critic.pth"):
            self.critic = torch.load(path + "_critic.pth")

