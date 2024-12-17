import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import torch
from torch.nn.functional import softmax

# Load the dataset
df = pd.read_csv('dataset/sentiment140.csv', encoding='latin1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
print(f"Unique values in target column before mapping: {df['target'].unique()}")

# Map sentiment labels
df['sentiment'] = df['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
df = df[['text', 'sentiment']]  # Retain only text and sentiment columns

# Sample 1% of data for each sentiment class
df_sampled = df.groupby('sentiment', group_keys=False).apply(lambda x: x.sample(frac=0.001, random_state=42)).reset_index(drop=True)

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_sampled['text'], df_sampled['sentiment'], test_size=0.2, random_state=42)

# Map string labels to integers AFTER splitting
label_map = {"negative": 0, "neutral": 1, "positive": 2}
train_labels = [label_map[label] for label in train_labels]
test_labels = [label_map[label] for label in test_labels]

from transformers import RobertaTokenizer, RobertaForSequenceClassification, BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load tokenizer and model
# fine-tuned
model = RobertaForSequenceClassification.from_pretrained('./final_RoBERTa_model')
tokenizer = RobertaTokenizer.from_pretrained('./final_RoBERTa_model')
#model = BertForSequenceClassification.from_pretrained('./final_BERT_model')
#tokenizer = BertTokenizer.from_pretrained('./final_BERT_model')

# baseline
#tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
#model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=2)
#tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# Tokenize the test data
inputs = tokenizer(list(test_texts), truncation=True, padding=True, return_tensors="pt")

# Perform inference with the pre-trained model
with torch.no_grad():  # Disable gradient computation
    outputs = model(**inputs)

# Get predictions
logits = outputs.logits
predictions = torch.argmax(logits, dim=-1).tolist()

# Evaluate metrics
accuracy = accuracy_score(test_labels, predictions)
f1 = f1_score(test_labels, predictions, average='weighted')

# Print the metrics
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")