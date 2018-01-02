import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.options.mode.chained_assignment = None

stock_csv = "BTC-USD.csv"
outfile = "BTC-USD_alltargets.csv"
datefile = ".".join(outfile.split(".")[:-1]) + "_latest_date.txt"


def MACD(df, period1, period2, periodSignal):
    EMA1 = pd.DataFrame.ewm(df, span=period1).mean()
    EMA2 = pd.DataFrame.ewm(df, span=period2).mean()
    MACD = EMA1 - EMA2
    Signal = pd.DataFrame.ewm(MACD,periodSignal).mean()
    Histogram = MACD - Signal
    return Histogram


def stochastics_oscillator(df, period):
    l, h = pd.DataFrame.rolling(df, period).min(), pd.DataFrame.rolling(df, period).max()
    k = 100 * (df - l) / (h - l)
    return k


def ATR(df, period):
    df['H-L'] = abs(df['High'] - df['Low'])
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    TR = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    return TR.to_frame()


def adjust_csv():
    with open(outfile, "r") as f:
        lines = f.readlines()[1:]

    lines.insert(0, "Close,CloseTarget,MACD,MACDTarget,Stochastics,StochasticsTarget,ATR,ATRTarget\n")

    # Remove entries where stochastics could not be calculated
    new_lines = []
    for line in lines:
        for count, col in enumerate(line.split(",")):
            if count == 4:
                if col == "":
                    break
                else:
                    new_lines.append(line)

    with open(outfile, "w") as f:
        for line in new_lines:
            f.write(line)


def clear_null():
    with open(stock_csv, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if "null" not in line:
            new_lines.append(line)

    with open(stock_csv, "w") as f:
        for line in new_lines:
            f.write(line)


if __name__ == "__main__":
    clear_null()
    df = pd.read_csv(stock_csv, usecols=[0,1,2,3,4])

    latest_date = df['Date'][df.index[-1]]
    dfPrices = df['Close']
    dfPriceShift = dfPrices.shift(-1)
    macd = MACD(dfPrices, 12, 26, 9)
    macdShift = macd.shift(-1)
    stochastics = stochastics_oscillator(dfPrices, 14)
    stochasticsShift = stochastics.shift(-1)
    atr = ATR(df, 14)
    atrShift = atr.shift(-1)

    final_data = pd.concat([dfPrices,
                            dfPriceShift,
                            macd,
                            macdShift,
                            stochastics,
                            stochasticsShift,
                            atr,
                            atrShift
                            ], axis=1)

    final_data.to_csv(outfile, index=False)
    with open(datefile, "w") as f:
        f.write(latest_date)

    adjust_csv()
