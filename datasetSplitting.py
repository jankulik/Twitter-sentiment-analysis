from datasets import load_dataset
import pandas as pd

dataset = load_dataset('csv', data_files="data/cleaned_tweets.csv", split='train')
dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=1)

train_data = pd.DataFrame(dataset['train'])
test_data = pd.DataFrame(dataset['test'])

train_data.to_csv("data/train_data.csv", index=False)
test_data.to_csv("data/test_data.csv", index=False)
