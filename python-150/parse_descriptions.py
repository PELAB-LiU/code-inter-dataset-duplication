import json

data_path = 'data.jsonl'
data_out = 'data_parsed_nl.jsonl'


def parse_nl(nl):
    parsed = nl.split('  ')[0]
    if parsed.startswith("\'"):
        parsed = parsed[1:]
    if parsed.endswith("\'"):
        parsed = parsed[:-1]
    return parsed


all_data = []
with open(data_path, 'r') as file:
    for line in file:
        data = json.loads(line)
        data['nl_parsed'] = parse_nl(data['nl'])
        all_data.append(data)

with open(data_out, 'w') as f:
    for item in all_data:
        json.dump(item, f)
        f.write('\n')
