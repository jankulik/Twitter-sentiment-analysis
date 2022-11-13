import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt


def benchmark(indexDateStart, indexDateStop):
    stockWallet = np.zeros(prices.shape)
    usdWallet = np.array([100])

    for i in range(indexDateStop - indexDateStart):
        date = dates[indexDateStart + i]
        if date not in sentiment.index:
            break

        if i > 0:
            usdWallet = np.append(usdWallet, np.sum(stockWallet[i - 1] * prices.loc[date].to_numpy()))

        for j in range(stockWallet.shape[1]):
            stockWallet[i, j] = usdWallet[i] / stockWallet.shape[1] / prices.loc[date, prices.columns[j]]

    return usdWallet


def backtestAbsolute(numStocks, minTweetCount, indexDateStart, indexDateStop):
    stockWallet = np.zeros(prices.shape)
    usdWallet = np.array([100])

    for i in range(indexDateStop - indexDateStart):
        date = dates[indexDateStart + i]
        if date not in sentiment.index:
            break

        if i > 0:
            usdWallet = np.append(usdWallet, np.sum(stockWallet[i - 1] * prices.loc[date].to_numpy()))

        tweetCount = sentiment.loc[date].to_numpy()[1::2]
        sentimentScores = sentiment.loc[date].to_numpy()[::2]
        sentimentScores = np.where(tweetCount > minTweetCount, sentimentScores, 0)

        if np.count_nonzero(sentimentScores) < numStocks:
            numStocks = np.count_nonzero(sentimentScores)

        sortedTickers = np.argsort(-sentimentScores)

        for j in range(stockWallet.shape[1]):
            if np.where(sortedTickers == j)[0][0] < numStocks:
                stockWallet[i, j] = usdWallet[i] / numStocks / prices.loc[date, prices.columns[j]]

    return usdWallet


def backtestRelative(numStocks, minTweetCount, indexDateStart, indexDateStop):
    stockWallet = np.zeros(prices.shape)
    usdWallet = np.array([100, 100])

    for i in range(1, indexDateStop - indexDateStart):
        datePrevious = dates[indexDateStart + i - 1]
        date = dates[indexDateStart + i]
        if date not in sentiment.index:
            break

        if i > 1:
            usdWallet = np.append(usdWallet, np.sum(stockWallet[i - 1] * prices.loc[date].to_numpy()))

        if i > 0:
            tweetCountPrevious = sentiment.loc[datePrevious].to_numpy()[1::2]
            tweetCount = sentiment.loc[date].to_numpy()[1::2]
            sentimentScoresPrevious = sentiment.loc[datePrevious].to_numpy()[::2]
            sentimentScores = sentiment.loc[date].to_numpy()[::2]

            sentimentScores = np.divide(sentimentScores - sentimentScoresPrevious,
                                        sentimentScoresPrevious,
                                        out=np.zeros_like(sentimentScores - sentimentScoresPrevious),
                                        where=sentimentScoresPrevious != 0)
            sentimentScores = np.where(np.logical_and(tweetCountPrevious > minTweetCount, tweetCount > minTweetCount),
                                       sentimentScores, 0)

            if np.count_nonzero(sentimentScores) < numStocks:
                numStocks = np.count_nonzero(sentimentScores)

            sortedTickers = np.argsort(-sentimentScores)

            for j in range(stockWallet.shape[1]):
                if np.where(sortedTickers == j)[0][0] < numStocks:
                    stockWallet[i, j] = usdWallet[i] / numStocks / prices.loc[date, prices.columns[j]]

    return usdWallet


def optimise(numStockRange, minTweetCountRange, indexDateStart, indexDateStop, method):
    xdata = np.zeros(numStockRange * minTweetCountRange)
    ydata = np.zeros(numStockRange * minTweetCountRange)
    zdata = np.zeros(numStockRange * minTweetCountRange)

    bestNumStock = 0
    bestMinTweetCount = 0
    bestUsdWallet = 0

    for i in range(1, numStockRange + 1):
        for j in range(1, minTweetCountRange + 1):
            usdWallet = np.average(method(i, j, indexDateStart, indexDateStop))

            xdata[(i - 1) * numStockRange + j - 1] = j
            ydata[(i - 1) * numStockRange + j - 1] = i
            zdata[(i - 1) * numStockRange + j - 1] = usdWallet

            if usdWallet > bestUsdWallet:
                bestNumStock = i
                bestMinTweetCount = j
                bestUsdWallet = usdWallet

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(xdata, ydata, zdata - 100)
    ax.set_xlabel('Minimal number of tweets')
    ax.set_ylabel('Number of traded stocks')
    ax.set_zlabel('Revenue [%]')
    fig.tight_layout()
    plt.show()

    return bestNumStock, bestMinTweetCount, bestUsdWallet


sentiment = pd.read_csv("data/historical_sentiment.csv", index_col='Date')
prices = pd.read_csv("data/historical_prices.csv", index_col='Date')
sentiment = sentiment.iloc[::-1]
prices = prices.iloc[::-1]

dates = list(prices.index)

# print(optimise(30, 30, 0, 127, backtestAbsolute))
# print(optimise(30, 30, 0, 127, backtestRelative))

indexDateStart = 127
indexDateStop = 252
plotDates = list(prices.index)[indexDateStart:indexDateStop]
x = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in plotDates]

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=round((indexDateStop - indexDateStart) / 6)))

plt.plot(x, backtestAbsolute(3, 24, indexDateStart, indexDateStop) - 100, label="Absolute")
plt.plot(x, backtestRelative(1, 23, indexDateStart, indexDateStop) - 100, label="Relative")
plt.plot(x, benchmark(indexDateStart, indexDateStop) - 100, label="Benchmark")
plt.axhline(y=0, color='black', linestyle='--')

plt.legend(loc="lower left")
plt.xlabel('Date')
plt.ylabel('Revenue [%]')

plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.savefig('plots/revenue.pdf', bbox_inches='tight')
plt.show()
