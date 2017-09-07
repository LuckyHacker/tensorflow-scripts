label_path = "food-101/meta/labels.txt"
target_file = "data/food-101_label_map.pbtxt"

with open(label_path, "r") as f:
    labels = f.read().split("\n")[:-1]

with open(target_file, "a") as f:
    for i in range(len(labels)):
        f.write("item {\n")
        f.write("\tid: {}\n".format(i + 1))
        f.write("\tname: '{}'\n".format(labels[i]))
        f.write("}\n")
