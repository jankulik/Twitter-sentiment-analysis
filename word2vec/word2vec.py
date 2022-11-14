import pandas as pd
from gensim.models import Word2Vec, KeyedVectors

df = pd.read_csv("data/cleaned_tweets.csv")
data = [row.split(' ') for row in df['text']]

model = Word2Vec(data, min_count=3, vector_size=100, workers=3, window=5, sg=1, epochs=100)
word_vectors = model.wv
word_vectors.save("data/vectors.kv")

reloaded_word_vectors = KeyedVectors.load("data/vectors.kv")
print(reloaded_word_vectors)
