import numpy as np
import pandas as pd
import datetime as dt
import re
import snscrape.modules.twitter as sntwitter
import time
from roberta import getScore


def calculateSentiment(row):
    print(row['Symbol'])
    query = '$' + row[
        'Symbol'] + ' lang:en since:' + startDate + ' until:' + stopDate + ' -filter:links -filter:replies'
    scores = np.array([])
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if checkSymbols(re.findall(r'[$][A-Za-z]\S*', tweet.content)):
            scores = np.append(scores, getScore(tweet.content))

    if len(scores) > 5:
        row['Score'] = np.average(scores)
    else:
        row['Score'] = 0
    row['Tweets'] = len(scores)
    return row


def calculateSingleSentiment(symbol):
    query = '$' + symbol + ' lang:en since:' + startDate + ' until:' + stopDate + ' -filter:links -filter:replies'
    scores = np.array([])
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if len(scores) >= 100:
            break
        if checkSymbols(re.findall(r'[$][A-Za-z]\S*', tweet.content)):
            # print(tweet.content)
            # print(getScore(tweet.content))
            # print(tweet.user.username)
            # print(tweet.date)
            # print(tweet.date.timestamp())
            # print('Iteration:', i)
            # print('--------------------------------------------------')
            scores = np.append(scores, getScore(tweet.content))
    print('--------------------------------------------------')
    print(scores)
    print(len(scores))
    return np.average(scores)


def checkSymbols(symbols):
    check = True
    for symbol in symbols:
        if symbols[0] != symbol:
            check = False
            break
    return check


start = time.time()

df = pd.read_csv("data/S&P100_symbols.csv")
df['Score'] = np.zeros(df.shape[0])
df['Tweets'] = np.zeros(df.shape[0])

# startDate = dt.date.today() - dt.timedelta(days=int(7))
# stopDate = startDate + dt.timedelta(days=int(1))
# startDate = startDate.strftime('%Y-%m-%d')
# stopDate = stopDate.strftime('%Y-%m-%d')

# df = df.apply(calculateSentiment, axis=1)
# df.to_csv("data/sentiment.csv", index=False)

# print(calculateSingleSentiment('AAPL'))

for i in range(20):
    startDate = dt.date.today() - dt.timedelta(days=int(i + 3))
    stopDate = startDate + dt.timedelta(days=int(1))
    startDate = startDate.strftime('%Y-%m-%d')
    stopDate = stopDate.strftime('%Y-%m-%d')

    print(calculateSingleSentiment('CVX'))

end = time.time()
print('Time elapsed:', round((end - start) / 60, 2))
