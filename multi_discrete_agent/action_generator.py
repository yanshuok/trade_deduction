import json
from copy import deepcopy
import re

actions = {"chn": [], "usa": []}
with open("actions.json", "r", encoding="utf8") as f:
    raw_actions = json.load(f)

for k, v in raw_actions.items():
    for x in v:
        d = []
        for i in x:
            if 'range' in i:
                for g in range(i['range'][0], i['range'][1] + i['step'], i['step']):
                    c = deepcopy(i)
                    c['default_rate'] = [g]
                    c['node_desc'] = re.sub(r"\[\d+\]", str(g), c['node_desc'])
                    d.append(c)
            else:
                d.append(i)
        actions[k].append(d)
print([len(x) for x in actions['chn']], [len(x) for x in actions['usa']])
with open("actions2.json", "w", encoding="utf8") as f:
    json.dump(actions, f, indent=2, ensure_ascii=False)