import requests
import numpy as np
import copy
import hashlib
import pickle
from pprint import pprint
import json


class Env:

    def __init__(self, server_addr, num_threads=8):
        self.server_addr = server_addr#"http://192.168.3.46:30001/"
        self.state_dim = 47488
        self.round = 1
        self.ts = 1
        self.headers = {"accept": "application/json"}
        self.actions = {'usa': [], 'chn': []}
        self.action_dim = {'usa': [], 'chn': []}
        self.action_index = {}
        self.minmax = {}
        self.init_action()
        self.state_uuid = ""
        self.total_steps = 0
        self.history_actions = {"usa": [], "chn": []}
        with open("../normalize_meta_data.json", "r") as f:
            self.normalize_meta_data = json.load(f)

    def init_action(self):
        with open("actions2.json", "r", encoding="utf8") as f:
            for k, v in json.load(f).items():
                for i in v:
                    self.actions[k].append(i)
                    self.action_dim[k].append(len(i))

    def formate_state(self, states):
        state_vec = []
        for state in states:
            s = np.array([float(x['Value']) for x in state['header_data']['data']])
            n = self.normalize_meta_data[state['header']]
            if n['std'] == 0:
                continue
            else:
                state_vec.append((s - n['mean']) / n['std'])
        return np.hstack(state_vec)[np.newaxis, :]

    def formate_reward(self, r):
        # with open(f"reward_data/reward_{self.total_steps}.pkl", "wb") as f:
        #     pickle.dump(r, f)
        reward = {}
        for idx, reg_name in r['name'].items():
            reward[reg_name.lower().strip()] = {}
            for k, v in r.items():
                if k != 'name':
                    reward[reg_name.lower().strip()][k] = v[idx]
        return reward

    def reset(self):
        self.history_actions = {"usa": [], "chn": []}
        self.round = 0
        self.ts = 0
        res = requests.post(f"{self.server_addr}reset?model_name=gtapXP", headers=self.headers)
        data = res.json()
        self.state_uuid = data['res']['state_uuid']
        return self.formate_state(data['res']['state'])

    def get_available_actions(self, region):
        available_actions = [0 for _ in range(len(self.actions[region]))]
        if self.round == 4 and region == "chn":
            available_actions[self.history_actions['usa'][-1]] = 1
        else:
            for i, row in enumerate(self.actions[region]):
                if row['round'] == self.round:
                    available_actions[i] = 1
        return available_actions

    def step(self, action, region):
        self.total_steps += 1
        action_entity = self.actions[region][self.round][action[self.round]]
        if 0 < self.round < 3:
            if self.history_actions[region][-1]['default_rate'][0] == 0:
                action_entity['other_info'] = None
            else:
                action_entity['other_info'] = self.history_actions[region][-1]['default_rate']
        params = {"state_uuid": self.state_uuid, "action": [action_entity]}
        # print(params)
        print(self.round, region, action, self.actions[region][self.round][action[self.round]]['node_desc'])
        self.history_actions[region].append(action_entity)
        res = requests.post(f"{self.server_addr}step", headers=self.headers, json=params)
        data = res.json()
        # print(data)
        self.state_uuid = data['res']['next_state_uuid']
        state = self.formate_state(data['res']['next_state'])
        reward = self.formate_reward(data['res']['reward'])
        self.ts += 1
        if self.ts > 11:
            done = True
        else:
            done = False
        if self.ts % 2 == 0:
            self.round += 1
        return state, {'chn': reward['chn']['vgdp'], 'usa': reward['usa']['vgdp']}, done

    def save(self):
        with open("minmax.pkl", "wb") as f:
            pickle.dump(self.minmax, f)


