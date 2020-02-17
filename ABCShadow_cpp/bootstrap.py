# coding: utf-8
#
# Copyright (c) 2020 quentinl-c.


import argparse
import sys
import json
from multiprocessing import Pool
from os import cpu_count
from pathlib import Path
import subprocess
# Default value
CPU_COUNT = min(cpu_count(), 25)
CMD = "./shadow"
ACTIONS = ["abc_estim", "gibbs_sim", "mh_sim"]

key_select = lambda x: "y_obs" if x == ACTIONS[0] else "theta"

def exec_shadow(action_idx, config_file, name, y_obs, size):
    args = [ACTIONS[action_idx], config_file, name, ','.join(map(str, y_obs)), *map(str,size)]
    cmd = [CMD]
    cmd.extend(args)
    print(cmd)
    p = subprocess.Popen(cmd)
    p.wait()
    print(f"{name} ended with returncode {p.returncode}")

   
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('action', type=str)
    parser.add_argument('input_file', type=str)
    parser.add_argument('config_file', type=str)
    parser.add_argument('output_dir', type=str)

    args = parser.parse_args()

    action = args.action 
    if action not in ACTIONS:
        print("Not the right action ...")
        sys.exit(1)

    input_file = Path(args.input_file)
    config_file = Path(args.config_file)

    if not input_file.is_file():
        raise ValueError("Invalid input file !")

    if not config_file.is_file():
        raise ValueError("Invalide config file !")

    # Ensure output directory exists
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(input_file, 'r') as json_f:
        teams = json.load(json_f)

    params = [(ACTIONS.index(action), args.config_file, str(output_dir / name), team[key_select(action)], team["size"]) for name, team in teams.items()]
    pool = Pool(min(CPU_COUNT, len(params)))
    pool.starmap(exec_shadow, params)
