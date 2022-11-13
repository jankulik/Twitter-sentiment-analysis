import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt

sentiment = pd.read_csv("data/historical_sentiment.csv", index_col='Date')
prices = pd.read_csv("data/historical_prices.csv", index_col='Date')


def plotSentimentVsPrice(ticker, days):
    dates = list(prices.index)[:days]
    x = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

    print("Correlation coefficient:", np.corrcoef(np.array(sentiment[ticker])[:len(dates)],
                                                  np.array(prices[ticker])[:len(dates)])[0, 1])

    fig, ax1 = plt.subplots()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=round(days / 6)))

    color = 'tab:orange'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Sentiment score [-]', color=color)
    ax1.plot(x, np.array(sentiment[ticker])[:len(dates)], color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim([0, 1.05])

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Price [USD]', color=color)
    ax2.plot(x, np.array(prices[ticker])[:len(dates)], color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig('plots/sentimentVsPrice.pdf', bbox_inches='tight')
    plt.show()


def plotSentiments(tickers, days):
    dates = list(prices.index)[:days]
    x = [dt.datetime.strptime(d, '%Y-%m-%d').date() for d in dates]

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=round(days / 6)))

    for ticker in tickers:
        plt.plot(x, np.array(sentiment[ticker])[:len(dates)], label=ticker)

    plt.legend(loc="upper left")
    plt.xlabel('Date')
    plt.ylabel('Sentiment score [-]')

    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.savefig('plots/sentiment.pdf', bbox_inches='tight')
    plt.show()


days = 43
plotSentimentVsPrice('AAPL', days)
plotSentiments(['META', 'TSLA'], days)
