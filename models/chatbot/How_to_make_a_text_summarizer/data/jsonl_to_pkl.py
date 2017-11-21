import json
import pickle

master_dict = {"desc": [], "head": []}
with open("sample-1M.jsonl", "r") as f:
    for count, line in enumerate(f.readlines()):
        print("{} dict out of {}".format(count+1, 1000000))
        d = json.loads(line)
        master_dict["desc"].append(d["content"])
        master_dict["head"].append(d["title"])

with open("tokens.pkl", "wb") as f:
    pickle.dump(master_dict, f)
