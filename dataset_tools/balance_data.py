import pandas_ml as pdml
import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("csv_path", nargs='?', default="None")
parser.add_argument("output_path", nargs='?', default="None")
parser.add_argument("column_name", nargs='?', default="None")
parser.add_argument("--sampling", nargs='?', default="undersample")
args = parser.parse_args()

def undersample(df):
    sampler = df.imbalance.under_sampling.ClusterCentroids()
    sampled = df.fit_sample(sampler)
    return sampled


def oversample(df):
    sampler = df.imbalance.over_sampling.SMOTE()
    sampled = df.fit_sample(sampler)
    return sampled


if __name__ == "__main__":
    if args.csv_path == "None":
        parser.print_help()

    if args.output_path == "None":
        parser.print_help()

    if args.column_name == "None":
        parser.print_help()

    if args.sampling not in ("undersample", "oversample"):
        parser.print_help()

    df = pd.read_csv(args.csv_path)
    df.columns = [".target" if x == args.column_name else x for x in df.columns]
    df = pdml.ModelFrame(df)
    print(df.target.value_counts())

    if args.sampling == "undersample":
        df = undersample(df)
    elif args.sampling == "oversample":
        df = oversample(df)

    df.columns = [args.column_name if x == ".target" else x for x in df.columns]
    df.to_csv(args.output_path, mode='w', header=True)
