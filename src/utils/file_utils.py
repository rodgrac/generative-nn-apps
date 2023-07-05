import json


def load_json(file_path):
    with open(file_path, 'r') as fp:
        return json.load(fp)