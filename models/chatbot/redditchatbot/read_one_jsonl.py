import json
import sys
import pprint


filepath = sys.argv[1]

with open(filepath, "rb") as f:
    json_data = str(f.readline(), "latin-1")


jsonl_dict = json.loads(json_data)

pp = pprint.PrettyPrinter()

pp.pprint(jsonl_dict)
