import requests
import numpy as np
import copy
import hashlib
import pickle
from pprint import pprint


class Env:

    def __init__(self, server_addr, num_threads=8):
        self.server_addr = server_addr#"http://192.168.3.46:30001/"
        self.state_dim = 17830
        self.round = 1
        self.ts = 1
        self.headers = {"accept": "application/json"}
        self.actions = {'usa': [], 'chn': []}
        self.action_index = {}
        self.minmax = {}
        self.init_action()
        self.state_uuid = ""

    def init_action(self):
        for rnd in range(1, 4):
            res = requests.post(f"{self.server_addr}get_action_space", headers=self.headers, params={'rnd': rnd})
            for reg in ['chn', 'usa']:
                for x in res.json()['res']['action_space'][reg]:
                    m = hashlib.md5()
                    m.update(" ".join([i['node_desc'] + i['initiator'] for i in x]).encode("utf8"))
                    idx = m.hexdigest()
                    if idx not in self.action_index:
                        self.action_index[idx] = len(self.actions[reg])
                        self.actions[reg].append(x)

    def formate_state(self, states):
        state_vec = []
        for state in states:
            s = np.array([float(x['Value']) for x in state['header_data']['data']])
            if state['header'] not in self.minmax:
                self.minmax[state['header']] = {}
                self.minmax[state['header']]['max'] = np.max(s)
                self.minmax[state['header']]['min'] = np.min(s)
            else:
                if np.max(s) > self.minmax[state['header']]['max']:
                    self.minmax[state['header']]['max'] = np.max(s)
                if np.min(s) < self.minmax[state['header']]['min']:
                    self.minmax[state['header']]['min'] = np.min(s)
            max_value = self.minmax[state['header']]['max']
            min_value = self.minmax[state['header']]['min']
            if max_value == 0 and min_value == 0:
                std = s
            elif max_value == min_value:
                std = s / max_value
            else:
                std = (s - min_value) / (max_value - min_value)
            state_vec.append(std)
        return np.hstack(state_vec)[np.newaxis, :]

    def formate_reward(self, r):
        reward = {}
        for idx, reg_name in r['name'].items():
            reward[reg_name.lower().strip()] = {}
            for k, v in r.items():
                if k != 'name':
                    reward[reg_name.lower().strip()][k] = v[idx]
        return reward

    def reset(self):
        self.round = 1
        self.ts = 0
        res = requests.post(f"{self.server_addr}reset?model_name=gtapXP", headers=self.headers)
        data = res.json()
        self.state_uuid = data['res']['state_uuid']
        return self.formate_state(data['res']['state'])

    def get_available_actions(self, region):
        res = requests.post(f"{self.server_addr}get_action_space", headers=self.headers, params={'rnd': self.round})
        available_actions = [0 for _ in range(len(self.actions[region]))]
        for x in res.json()['res']['action_space'][region]:
            m = hashlib.md5()
            m.update(" ".join([i['node_desc'] + i['initiator'] for i in x]).encode("utf8"))
            idx = m.hexdigest()
            available_actions[self.action_index[idx]] = 1
        return available_actions

    def step(self, action, region):
        params = {"state_uuid": self.state_uuid, "action": self.actions[region][action]}
        print(region, action, ",".join([x['node_desc'] for x in self.actions[region][action]]))
        res = requests.post(f"{self.server_addr}step", headers=self.headers, json=params)
        data = res.json()
        self.state_uuid = data['res']['next_state_uuid']
        state = self.formate_state(data['res']['next_state'])
        reward = self.formate_reward(data['res']['reward'])

        self.ts += 1
        if self.ts > 3:
            done = True
        else:
            done = False
        if self.ts % 2 == 0:
            self.round += 1
        return state, reward[region]['vgdp'], done

    def save(self):
        with open("minmax.pkl", "wb") as f:
            pickle.dump(self.minmax, f)

