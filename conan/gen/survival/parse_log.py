import os
import pathlib
import json

log_path = os.path.join(pathlib.Path(__file__).parent, "../save/survival")
config_path = pathlib.Path(__file__).parent

print(log_path)

with open(os.path.join(config_path, "gen.log"), "r") as f:
    logs = f.readlines()

traces = os.listdir(log_path)
num = len(traces)

id = -1
for line in logs:
    line = line.strip()
    if line.startswith("Survival flag"):
        id += 1
        if id > 0:
            if os.path.isdir(os.path.join(log_path, f"survival_{id-1}")):
                json.dump(config, open(os.path.join(log_path, f"survival_{id-1}/survival_config.json"), "w"))
        config = {"survival flag": None, "traces": [], "died": False, "cause": "none"}
        survival_flag = line.split(":")[-1]
        config["survival flag"] = survival_flag
        print(f"----- survival trace {id} {survival_flag}-----")
    if line.startswith("--- "):
        config["traces"].append(line.split("--- ")[-1].split(" ---")[0])
    if line.startswith("Died of "):
        config["died"] = True
        config["cause"] = line.split("Died of ")[-1].strip()
    if line.startswith("Get") or line.startswith("Drink") or line.startswith("Eat") or line.startswith("Place") or line.startswith("Sleep"):
        config["traces"].append(line)
    if "health 0" in line:
        print("parse hurt")
        config["died"] = True
        config["cause"] = "hurt"

if os.path.isdir(os.path.join(log_path, f"survival_{id}")):
    json.dump(config, open(os.path.join(log_path, f"survival_{id}/survival_config.json"), "w"))
