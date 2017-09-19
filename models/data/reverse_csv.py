import pandas as pd

csv = "BTCUSD.csv"
outfile = "reversed_" + csv

df = pd.read_csv(csv)
df = df.iloc[::-1]
df.to_csv(outfile, index=False)
