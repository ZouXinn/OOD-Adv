# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import collections

import json
import os

import tqdm

from domainbed.lib.query import Q

def load_records(path):
    records = []
    for i, subdir in tqdm.tqdm(list(enumerate(os.listdir(path))),
                               ncols=80,
                               leave=False): # 
        results_path = os.path.join(path, subdir, "results.jsonl") # the path of the results.jsonl file
        try:
            with open(results_path, "r") as f:
                for line in f:
                    records.append(json.loads(line[:-1])) # each line/record is the result of a checkpoint_freq in the train.py file
        except IOError:
            pass

    return Q(records)

def get_grouped_records(records):
    """Group records by (trial_seed, dataset, algorithm, test_env). Because
    records can have multiple test envs, a given record may appear in more than
    one group."""
    result = collections.defaultdict(lambda: [])
    for r in records:
        for test_env in r["args"]["test_envs"]: # a record may include results for many test envs
            group = (r["args"]["trial_seed"],
                r["args"]["dataset"],
                r["args"]["algorithm"],
                test_env)
            result[group].append(r) # the group is the index and r is the corresponding record
    return Q([{"trial_seed": t, "dataset": d, "algorithm": a, "test_env": e,
        "records": Q(r)} for (t,d,a,e),r in result.items()]) # return the list of dictionaries {"trial_seed": ..., "dataset": ..., "algorithm": ..., "test_env": ..., "records": Q(r)}
