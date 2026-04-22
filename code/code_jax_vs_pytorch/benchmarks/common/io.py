import json
import os
import pickle


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_pickle(path, payload):
    with open(path, "wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)
