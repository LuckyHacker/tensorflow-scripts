import pandas as pd


df = pd.read_csv("creditcard_sampled.csv")
df.loc[df.Fraud == 0, 'Normal'] = 1
df.loc[df.Fraud == 1, 'Normal'] = 0

df.to_csv("creditcard.csv", index=False)
