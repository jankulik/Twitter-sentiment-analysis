from transformers import AutoModelForSequenceClassification
from transformers import AutoTokenizer, AutoConfig
import numpy as np
from scipy.special import softmax
import re


def preprocess(text):
    text = re.sub(r'https?\S+', 'http', text)
    text = re.sub(r'@\S+', '@user', text)
    text = re.sub(r'\d+', '@number', text)
    text = re.sub(r'[$][A-Za-z]\S*', '@stock', text)
    text = re.sub('[^A-Za-z0-9@]', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.lower()
    text = text.strip()
    return text


def getScore(text):
    text = preprocess(text)
    encoded_input = tokenizer(text, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores[1] = 0
    scores = scores / np.sum(scores)
    return scores[2]


MODEL = f"tuned_model"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
