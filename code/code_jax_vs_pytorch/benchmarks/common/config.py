import os

import yaml


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_run_dir(root_dir, backend, benchmark_name, seed):
    return os.path.join(root_dir, backend, benchmark_name, "seed_{0}".format(seed))
