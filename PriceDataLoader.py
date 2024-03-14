
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import datetime
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
import plotly.graph_objs as go
import seaborn as sns


metrics = ['blocks-size', 'avg-block-size', 'n-transactions-total',
           'hash-rate', 'difficulty', 'transaction-fees-usd',
           'n-unique-addresses', 'n-transactions', 'my-wallet-n-users',
           'utxo-count', 'n-transactions-excluding-popular', 'estimated-transaction-volume-usd',
           'trade-volume', 'total-bitcoins', 'market-price']

years = [2020]

def loadPriceData(metrics , years : int):
    df_all = pd.DataFrame()
    for m in metrics:
        append_data = []
        for y in years:
            ts = datetime.datetime(
                y, 12, 31, tzinfo=datetime.timezone.utc).timestamp()
            print('https://api.blockchain.info/charts/'+m +
                '?timespan=1year&rollingAverage=24hours&format=csv&start='+str(int(ts)))
            df = pd.read_csv('https://api.blockchain.info/charts/'+m+'?timespan=1year&rollingAverage=24hours&format=csv&start='+str(
                int(ts)), names=['date', m], parse_dates=[0], index_col=[0])
            append_data.append(df)
        df_m = pd.concat(append_data)
        df_m.index = df_m.index.normalize()
        df_m = df_m.groupby([pd.Grouper(freq='D')]).mean()

        if df_all.shape[0] == 0:
            print(m)
            print(df_m.shape)
            df_all = df_m
        else:
            print(m)
            print(df_m.shape)
            print(df_all.shape)
            df_all = df_all.merge(df_m, on="date", how="outer")
    
    return df_all


# def cleanData(df_all):
#     print('Cleaning Data...')
#     df_all['date'] = df_all.index
#     df_all["price"] = df_all["market-price"]
#     df_all.drop(["market-price"], axis=1, inplace=True)
#     df_all[["next-price"]] = df_all[["price"]].shift(-1)
#     df_all.loc[(df_all["price"] <= df_all["next-price"]), "signal"] = 1
#     df_all.loc[(df_all["price"] >= df_all["next-price"]), "signal"] = 0
#     df_all.dropna(subset=["signal"], inplace=True)
#     df_all[["signal"]] = df_all[["signal"]].astype(int)
#     df_all.drop(["next-price"], axis=1, inplace=True)
#     df_all.dropna(inplace=True)
#     print('Renamed price and added signals')
#     return df_all

def cleanData(df_all):
    df_all['date'] = df_all.index
    df_all["price"] = df_all["market-price"]
    df_all.drop(["market-price"], axis=1, inplace=True)
    df_all['SMA10'] = df_all['price'].rolling(window=10, min_periods=1, center=False).mean()
    df_all['SMA50'] = df_all['price'].rolling(window=50, min_periods=1, center=False).mean()
    df_all['SMA100'] = df_all['price'].rolling(window=500, min_periods=1, center=False).mean()
    df_all['EMA10'] = df_all['price'].ewm(span=10).mean()
    df_all['EMA50'] = df_all['price'].ewm(span=50).mean()
    df_all['EMA100'] = df_all['price'].ewm(span=100).mean()
    df_all.dropna(inplace=True)
    print('Renamed price and added technical indicators')
    return df_all



def plotPrice(df):
    trace1 = go.Scatter(
    x=df.index,
    y=df['price'],
    mode='lines',
    name='Original Price'
            )
    layout = dict(
        title='<b>Bitcoin Price</b>',
        yaxis=dict(title='<b>BTC Price (USD)</b>')
    )
    pdata = [trace1]
    fig = dict(data=pdata, layout=layout)
    iplot(fig, filename="Time Series with Rangeslider")

def plotAllFeatures(df):
    df.plot(subplots=True,
        layout=(8, 3),
        figsize=(22,22),
        fontsize=10,
        linewidth=2,
        sharex=False,
        title='Visualization of Original Time Series')
    plt.show()

def plotCorr(df):
    corr_matrix = df.corr(method="spearman")
    f, ax = plt.subplots(figsize=(16,8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', linewidth=0.4,
                annot_kws={"size": 10}, cmap='coolwarm', ax=ax)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.show()




