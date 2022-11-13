import pandas as pd
import re
from nltk.corpus import words


def cleanText(text):
    text = re.sub(r'https?\S+', 'http', text)
    text = text.replace('user', '@user')
    text = re.sub(r'@\S+', '@user', text)
    text = re.sub(r'\d+', '@number', text)

    upperCase = re.findall(r'\b[A-Z]+(?:\s+[A-Z]+)*\b', text)
    for i in range(len(upperCase)):
        upperCaseSplit = upperCase[i].split()
        for j in range(len(upperCaseSplit)):
            if upperCaseSplit[j] in nasdaqSymbols or upperCaseSplit[j].lower() not in englishWords:
                text = text.replace(upperCaseSplit[j], '@stock')

    text = re.sub('[^A-Za-z0-9@]', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.lower()
    text = text.strip()
    return text


df = pd.read_csv("data/stock_tweets.csv")
nasdaq = pd.read_csv("data/nasdaq_stocks.csv")
nasdaqSymbols = nasdaq.iloc[:, 0].values
englishWords = set(words.words())

df['Text'] = df['Text'].apply(cleanText)

df = df.rename(columns={'Text': 'text', 'Sentiment': 'label'})
df.to_csv("data/cleaned_tweets.csv", index=False)
