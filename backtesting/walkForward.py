import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt


def benchmark(indexDateStart, indexDateStop):
    stockWallet = np.zeros(prices.shape)
    wealth = np.array([1])

    for i in range(indexDateStop - indexDateStart):
        date = dates[indexDateStart + i]
        if date not in sentiment.index:
            break

        if i > 0:
            wealth = np.append(wealth, np.sum(stockWallet[i - 1] * prices.loc[date].to_numpy()))

        for j in range(stockWallet.shape[1]):
            stockWallet[i, j] = wealth[i] / stockWallet.shape[1] / prices.loc[date, prices.columns[j]]

    return wealth


def backtestAbsolute(numStocks, minTweetCount, indexDateStart, indexDateStop, startWealth):
    stockWallet = np.zeros(prices.shape)
    wealth = np.array([startWealth])

    for i in range(indexDateStop - indexDateStart):
        date = dates[indexDateStart + i]
        if date not in sentiment.index:
            break

        if i > 0:
            wealth = np.append(wealth, np.sum(stockWallet[i - 1] * prices.loc[date].to_numpy()))

        tweetCount = sentiment.loc[date].to_numpy()[1::2]
        sentimentScores = sentiment.loc[date].to_numpy()[::2]
        sentimentScores = np.where(tweetCount > minTweetCount, sentimentScores, 0)

        if np.count_nonzero(sentimentScores) < numStocks:
            numStocks = np.count_nonzero(sentimentScores)

        sortedTickers = np.argsort(-sentimentScores)

        for j in range(stockWallet.shape[1]):
            if np.where(sortedTickers == j)[0][0] < numStocks:
                stockWallet[i, j] = wealth[i] / numStocks / prices.loc[date, prices.columns[j]]

    if indexDateStop < len(dates):
        finalWealth = np.sum(stockWallet[indexDateStop - indexDateStart - 1] * prices.loc[dates[indexDateStop]].to_numpy())
    else:
        finalWealth = 0
    return wealth, finalWealth


def backtestRelative(numStocks, minTweetCount, indexDateStart, indexDateStop, startWealth):
    stockWallet = np.zeros(prices.shape)
    wealth = np.array([startWealth, startWealth])

    for i in range(1, indexDateStop - indexDateStart):
        datePrevious = dates[indexDateStart + i - 1]
        date = dates[indexDateStart + i]
        if date not in sentiment.index:
            break

        if i > 1:
            wealth = np.append(wealth, np.sum(stockWallet[i - 1] * prices.loc[date].to_numpy()))

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
                    stockWallet[i, j] = wealth[i] / numStocks / prices.loc[date, prices.columns[j]]

    if indexDateStop < len(dates):
        finalWealth = np.sum(stockWallet[indexDateStop - indexDateStart - 1] * prices.loc[dates[indexDateStop]].to_numpy())
    else:
        finalWealth = 0
    return wealth, finalWealth


def optimise(numStockRange, minTweetCountRange, indexDateStart, indexDateStop, method):
    xdata = np.zeros(numStockRange * minTweetCountRange)
    ydata = np.zeros(numStockRange * minTweetCountRange)
    zdata = np.zeros(numStockRange * minTweetCountRange)

    bestNumStock = 0
    bestMinTweetCount = 0
    bestWealth = 0

    for i in range(1, numStockRange + 1):
        for j in range(1, minTweetCountRange + 1):
            wealth = np.average(method(i, j, indexDateStart, indexDateStop, 1)[0])

            xdata[(i - 1) * numStockRange + j - 1] = j
            ydata[(i - 1) * numStockRange + j - 1] = i
            zdata[(i - 1) * numStockRange + j - 1] = wealth

            if wealth > bestWealth:
                bestNumStock = i
                bestMinTweetCount = j
                bestWealth = wealth

    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(xdata, ydata, zdata - 1)
    # ax.set_xlabel('Minimal number of tweets')
    # ax.set_ylabel('Number of traded stocks')
    # ax.set_zlabel('Revenue [%]')
    # fig.tight_layout()
    # plt.show()

    return bestNumStock, bestMinTweetCount, bestWealth


sentiment = pd.read_csv("data/historical_sentiment.csv", index_col='Date')
prices = pd.read_csv("data/historical_prices.csv", index_col='Date')
sentiment = sentiment.iloc[::-1]
prices = prices.iloc[::-1]

dates = list(prices.index)

split = 0.8
folds = 10
lenInSample = round(len(dates) * split / folds / (1 - split + split / folds))
lenOutSample = round(lenInSample * (1 - split) / split)

wealthAbsolute = np.array([])
wealthRelative = np.array([])
finalWealthAbsolute = 1
finalWealthRelative = 1
for i in range(folds):
    print("Progress: " + str(round(i / folds * 100, 2)) + "%")

    numStockAbsolute, minTweetCountAbsolute = optimise(20,
                                                       60,
                                                       lenOutSample * i,
                                                       lenOutSample * i + lenInSample,
                                                       backtestAbsolute)[:2]
    numStockRelative, minTweetCountRelative = optimise(20,
                                                       60,
                                                       lenOutSample * i,
                                                       lenOutSample * i + lenInSample,
                                                       backtestRelative)[:2]

    appendedWealthAbsolute, finalWealthAbsolute = backtestAbsolute(numStockAbsolute,
                                                                   minTweetCountAbsolute,
                                                                   lenOutSample * i + lenInSample,
                                                                   lenOutSample * i + lenInSample + lenOutSample,
                                                                   finalWealthAbsolute)

    appendedWealthRelative, finalWealthRelative = backtestRelative(numStockRelative,
                                                                   minTweetCountRelative,
                                                                   lenOutSample * i + lenInSample,
                                                                   lenOutSample * i + lenInSample + lenOutSample,
                                                                   finalWealthRelative)

    wealthAbsolute = np.hstack((wealthAbsolute, appendedWealthAbsolute))
    wealthRelative = np.hstack((wealthRelative, appendedWealthRelative))

plotDates = list(prices.index)[lenInSample:]
x = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in plotDates]
wealthBenchmark = benchmark(lenInSample, len(dates))

dailyReturnsAbsolute = np.append([0], np.diff(wealthAbsolute) / wealthAbsolute[:-1])
dailyReturnsRelative = np.append([0], np.diff(wealthRelative) / wealthRelative[:-1])
dailyReturnsBenchmark = np.append([0], np.diff(wealthBenchmark) / wealthBenchmark[:-1])

print("Cumulative return absolute [%]:", (wealthAbsolute[-1] - 1) * 100)
print("Cumulative return relative [%]:", (wealthRelative[-1] - 1) * 100)
print("Cumulative return benchmark [%]:", (wealthBenchmark[-1] - 1) * 100, '\n')

print("Annualised return absolute [%]:", (wealthAbsolute[-1] ** (252 / len(x)) - 1) * 100)
print("Annualised return relative [%]:", (wealthRelative[-1] ** (252 / len(x)) - 1) * 100)
print("Annualised return benchmark [%]:", (wealthBenchmark[-1] ** (252 / len(x)) - 1) * 100, '\n')

print("Annualised volatility absolute [%]:", np.std(dailyReturnsAbsolute) * np.sqrt(252) * 100)
print("Annualised volatility relative [%]:", np.std(dailyReturnsRelative) * np.sqrt(252) * 100)
print("Annualised volatility benchmark [%]:", np.std(dailyReturnsBenchmark) * np.sqrt(252) * 100, '\n')

riskFreeRate = 0.0421 / 10 / 252
print("Sharpe ratio absolute:",
      np.sqrt(252) * (np.average(dailyReturnsAbsolute) - riskFreeRate) / np.std(dailyReturnsAbsolute))
print("Sharpe ratio relative:",
      np.sqrt(252) * (np.average(dailyReturnsRelative) - riskFreeRate) / np.std(dailyReturnsRelative))
print("Sharpe ratio benchmark:",
      np.sqrt(252) * (np.average(dailyReturnsBenchmark) - riskFreeRate) / np.std(dailyReturnsBenchmark), '\n')

print("Maximum drowdown absolute [%]:",
      (np.min(wealthAbsolute) - np.max(wealthAbsolute)) / np.max(wealthAbsolute) * 100)
print("Maximum drowdown relative [%]:",
      (np.min(wealthRelative) - np.max(wealthRelative)) / np.max(wealthRelative) * 100)
print("Maximum drowdown benchmark [%]:",
      (np.min(wealthBenchmark) - np.max(wealthBenchmark)) / np.max(wealthBenchmark) * 100)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=round((len(dates) - lenOutSample) / 7)))

plt.plot(x, (wealthAbsolute - 1) * 100, label="Absolute")
plt.plot(x, (wealthRelative - 1) * 100, label="Relative")
plt.plot(x, (benchmark(lenInSample, len(dates)) - 1) * 100, label="Benchmark")
plt.axhline(y=0, color='black', linestyle='--')

# for i in range(folds):
#     plt.axvline(x=x[lenOutSample * i], color='b')

plt.legend(loc="lower left")
plt.xlabel('Date')
plt.ylabel('Revenue [%]')

plt.tight_layout()
plt.gcf().autofmt_xdate()
plt.savefig('backtesting/revenue.pdf', bbox_inches='tight')
plt.show()
