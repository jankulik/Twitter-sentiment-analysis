from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer
import pandas as pd


def adjustLabels(label):
    return label + 1


def findMaxLen(dataset):
    maxLen = 0
    for sequence in dataset:
        if len(sequence.split()) > maxLen:
            maxLen = len(sequence.split())
    return maxLen


def tokenizeFunction(data):
    return tokenizer(data['text'], padding="max_length", truncation=True, max_length=maxLen)


def computeMetrics(eval_pred):
    logits, labels = eval_pred
    # setting up all logits corresponding to neutral to nan in order to filter them out
    logits[:, 1] = np.nan
    predictions = np.nanargmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# adjusting labels to 0 → negative and 2 → positive
df = pd.read_csv("data/cleaned_labelled_tweets.csv")
df['label'] = df['label'].apply(adjustLabels)
df.to_csv("data/tweets_roberta.csv", index=False)

dataset = load_dataset('csv', data_files="data/tweets_roberta.csv", split='train')
maxLen = findMaxLen(dataset['text'])

dataset = dataset.train_test_split(test_size=0.2, shuffle=True, seed=1)
tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
tokenized_datasets = dataset.map(tokenizeFunction, batched=True)

train_dataset = tokenized_datasets['train']
eval_dataset = tokenized_datasets['test']

model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
training_args = TrainingArguments(output_dir="transformers/training_checkpoints", evaluation_strategy="epoch")

metric = evaluate.load('accuracy')

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=computeMetrics,
)

trainer.train()
trainer.save_model('tuned_model')
tokenizer.save_pretrained('tuned_model')
