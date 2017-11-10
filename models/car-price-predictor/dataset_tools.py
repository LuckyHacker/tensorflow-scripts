import pandas_ml as pdml
import pandas as pd
import numpy as np
import json
from keras.preprocessing.text import Tokenizer

class DeNormalizer:

    def __init__(self, dataframe, column):
        self.multiplier = (dataframe[column].max() - dataframe[column].min()) + dataframe[column].min()

    def convert_scalar(self, scalar):
        return scalar * self.multiplier


def normalize(dataframe):
    return (dataframe - dataframe.min()) / (dataframe.max() - dataframe.min())


class SplitDataframe:

    def __init__(self, dataframe, train_size=0.8):
        self.dataframe = dataframe
        msk = np.random.rand(len(self.dataframe)) < train_size
        self.traindata = self.dataframe[msk]
        self.testdata = self.dataframe[~msk]


class Balancer:

    def __init__(self, dataframe, label_column, undersample=True, oversample=False):
        self.df = pdml.ModelFrame(dataframe)
        self.df.columns = [".target" if x == label_column else x for x in self.df.columns]
        if undersample:
            self._undersample()
        if oversample:
            self._oversample()


    def _undersample(self):
        sampler = self.df.imbalance.under_sampling.ClusterCentroids()
        self.undersampled = self.df.fit_sample(sampler)
        self.undersampled.columns = [args.column_name if x == ".target" else x for x in self.df.columns]


    def _oversample(self):
        sampler = self.df.imbalance.over_sampling.SMOTE()
        self.oversampled = self.df.fit_sample(sampler)
        self.oversampled.columns = [args.column_name if x == ".target" else x for x in self.df.columns]


def tokenize_dataframe(df, token_folder):

    def dump_tokens_json(d, column):
        json_string = json.dumps(d, sort_keys=True, indent=4)
        with open("{}/{}.json".format(token_folder, column), "w") as f:
            f.write(json_string)

    for column in df:
        if any(isinstance(x, str) for x in df[column]):
            token = Tokenizer(num_words=None, filters='\t\n',
                                lower=False, split=",", char_level=False)
            token.fit_on_texts(df[column])
            if "nan" in token.word_index:
                print(token.word_index)
                input()
            df[column] = df[column].map(token.word_index)
            dump_tokens_json(token.word_index, column)

    return df
