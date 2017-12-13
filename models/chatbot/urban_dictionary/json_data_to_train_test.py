import json


json_path = "quotes.json"
train_test_ratio = 0.9

def read_json_to_dict(path):
    with open(path, "r") as f:
        json_string = f.read()

    return json.loads(json_string)


def dict_to_list(d):
    master_list = []
    for word in d:
        for x_y in d[word]:
            master_list.append(x_y)

    return master_list


def split_train_test(l, train_test_ratio):
    train_list = l[:int(len(l) * train_test_ratio)]
    test_list = l[int(len(l) * train_test_ratio):]
    return train_list, test_list


def save_train_test_lists(train_list, test_list):
    with open("train.from", "w") as ffrom:
        with open("train.to", "w") as fto:
            for tpl in train_list:
                ffrom.write(tpl[0] + "\n")
                fto.write(tpl[1] + "\n")

    with open("test.from", "w") as ffrom:
        with open("test.to", "w") as fto:
            for tpl in test_list:
                ffrom.write(tpl[0] + "\n")
                fto.write(tpl[1] + "\n")

if __name__ == "__main__":
    quotes = read_json_to_dict(json_path)
    quotes = dict_to_list(quotes)
    train_quotes, test_quotes = split_train_test(quotes, train_test_ratio)
    save_train_test_lists(train_quotes, test_quotes)
