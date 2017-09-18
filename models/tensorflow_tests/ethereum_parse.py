import matplotlib.pyplot as plt
from datetime import datetime
from collections import OrderedDict
import json
import itertools
import operator
import csv

dataset = "data/ethereum.json"

with open(dataset, "r") as f:
    json_data = json.loads(f.read())

data = list(map(lambda x: (x["time"].split("T")[0], x["usd"]), json_data["data"]))

date_list = []
for key, group in itertools.groupby(data, operator.itemgetter(0)):
    date_list.append(list(group))

avg_list = []
for date in date_list:
    avg_list.append((date[0][0], sum(i for _, i in date) / len(date)))

x, y = list(map(list, zip(*avg_list)))

xr = list(range(len(x)))

plt.figure(figsize = (18,7))
plt.plot(xr, y)
plt.title('Ethereum Stock History')
plt.legend(loc = 'upper left')
plt.show()


ethereum_dict = OrderedDict({   "Date": x,
                                "Price (USD)": y,
                                })
"""
with open("data/ethereum.csv", "w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(ethereum_dict.keys())
    writer.writerows(zip(*ethereum_dict.values()))
"""
