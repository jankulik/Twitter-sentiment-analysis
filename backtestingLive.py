import pandas as pd
import datetime as dt
import numpy as np
import snscrape.modules.twitter as sntwitter
import re
from roberta import getScore


def checkSymbols(symbols):
    check = True
    for symbol in symbols:
        if symbols[0] != symbol:
            check = False
            break
    return check


# initial wallet value
wallet = 10000
# max number of assets
numAssets = 10

df = pd.read_csv("data/historical_prices.csv")

tickers = list(df.columns)
tickers.pop(0)
for i in range(len(df)):
    stopDate = dt.datetime.strptime(df.loc[i, 'Date'], '%Y-%m-%d') + dt.timedelta(days=int(1))
    stopDate = stopDate.strftime('%Y-%m-%d')
    startDate = df.loc[i, 'Date']

    if i != 0:
        wallet = 0

    scores = np.zeros(len(tickers))
    for j in range(len(tickers)):
        query = '$' + tickers[j] + ' lang:en since:' + startDate + ' until:' + stopDate + ' -filter:links -filter:replies'
        tickerScores = np.array([])
        for k, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if checkSymbols(re.findall(r'[$][A-Za-z]\S*', tweet.content)):
                tickerScores = np.append(tickerScores, getScore(tweet.content))
        if len(tickerScores) > 5:
            scores[j] = np.average(tickerScores)
        else:
            scores[j] = 0
        if i != 0:
            wallet += df.loc[i - 1, tickers[j]] * df.loc[i, tickers[j]]

    sortedTickers = np.argsort(-scores)
    for j in range(len(tickers)):
        if np.where(sortedTickers == j)[0][0] < numAssets:
            df.loc[i, tickers[j]] = wallet / numAssets / df.loc[i, tickers[j]]
        else:
            df.loc[i, tickers[j]] = 0
    df.loc[i, 'Wallet'] = wallet
    print(df.loc[i, 'Date'] + ': $' + str(wallet) + ' Progress: ' + str(round(i / len(df) * 100, 2)) + '%')

df.to_csv("data/result.csv", index=False)
