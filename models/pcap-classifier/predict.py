from keras.models import model_from_json
from collections import Counter
from dataset_tools import normalize, SplitDataframe, Balancer, tokenize_dataframe
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


model_path = "model"
csv_file_path = "data/UNSW_NB15_testing-set.csv"

df = pd.read_csv(csv_file_path)

def prepare_data(df):
    df = tokenize_dataframe(df, "tokens")
    dataset_norm = normalize(df)

    x_test = dataset_norm.drop(["label", "attack_cat"], axis=1).as_matrix()
    y_test = dataset_norm[["label"]].as_matrix()

    return x_test, y_test


if __name__ == "__main__":
    x_test, y_test = prepare_data(df)

    with open("{}.json".format(model_path), "r") as f:
        model = model_from_json(f.read())

    model.load_weights("{}.h5".format(model_path))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    score = model.evaluate(x_test, y_test)
    print("\nAccuracy: {0:.2f} % ".format(score[1] * 100))
