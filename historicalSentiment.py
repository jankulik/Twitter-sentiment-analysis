import numpy as np
import pandas as pd
import datetime as dt
import re
import snscrape.modules.twitter as sntwitter
from joblib import Parallel, delayed
from roberta import getScore
import time


def checkSymbols(symbols):
    check = True
    for symbol in symbols:
        if symbols[0] != symbol:
            check = False
            break
    return check


def historicalSentiment(i):
    stopDate = dt.datetime.strptime(dates[i], '%Y-%m-%d') + dt.timedelta(days=int(1))
    stopDate = stopDate.strftime('%Y-%m-%d')
    startDate = dates[i]

    scores = np.zeros(2 * len(symbols))
    for j in range(len(symbols)):
        query = '$' + symbols[
            j] + ' lang:en since:' + startDate + ' until:' + stopDate + ' -filter:links -filter:replies'
        symbolScores = np.array([])
        for k, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
            if len(symbolScores) >= 100:
                break
            if checkSymbols(re.findall(r'[$][A-Za-z]\S*', tweet.content)):
                symbolScores = np.append(symbolScores, getScore(tweet.content))

        scores[2 * j] = np.average(symbolScores) if len(symbolScores) > 0 else 0
        scores[2 * j + 1] = len(symbolScores)

    return scores


symbols = pd.read_csv("data/S&P100_symbols.csv")['Symbol']

columns = []
for i in range(len(symbols)):
    columns.append(symbols[i])
    columns.append(symbols[i] + ' TweetCount')

date = dt.datetime.strptime('2022-02-02', '%Y-%m-%d')
dates = []
for i in range(365):
    if date.weekday() < 5:
        dates.append(date.strftime('%Y-%m-%d'))
    date = date - dt.timedelta(days=int(1))

start = time.time()

output = pd.DataFrame(columns=columns, index=dates)
results = np.array(Parallel(n_jobs=2)(delayed(historicalSentiment)(i) for i in range(2)))

output = pd.DataFrame(results, columns=columns, index=dates[:results.shape[0]])
output.index.name = 'Date'
output.to_csv("data/historical_sentiment.csv")

end = time.time()
print('Time elapsed:', round((end - start) / 60, 2))
