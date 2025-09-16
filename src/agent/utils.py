import json


def json_read(json_file):
    with open(json_file, "r") as f:
        return json.load(f)


def json_save(json_file, contents):
    with open(json_file, "w") as f:
        json.dump(contents, f, indent=2)