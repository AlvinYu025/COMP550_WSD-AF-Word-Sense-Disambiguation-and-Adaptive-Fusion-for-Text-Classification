import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import torch
from torch.nn.functional import softmax

# Load the dataset
df = pd.read_csv('dataset/amazon.csv', encoding='latin1', header=None)
df.columns = ['marketplace', 'customer_id', 'review_id', 'product_id', 'product_parent','product_title', 'product_category', 'star_rating', 'helpful_votes', 'total_votes', 'vine', 'verified_purchase', 'review_headline', 'review_body', 'review_date', 'sentiment']
# Remove rows where 'sentiment' column contains the string 'sentiment'
df = df[df['sentiment'] != 'sentiment']

# Convert '1' and '0' to integers
df['sentiment'] = df['sentiment'].astype(int)

# Map sentiment labels (0 -> 'negative', 1 -> 'positive')
df['sentiment'] = df['sentiment'].map({0: 'negative', 1: 'positive'})

# Check the sentiment class distribution after mapping
print(f"Sentiment class distribution after mapping: {df['sentiment'].value_counts()}")

# Retain only text (review_body) and sentiment columns
df = df[['review_body', 'sentiment']]

# Balance the dataset by sampling an equal number of positive and negative samples
df_balanced = df.groupby('sentiment').apply(lambda x: x.sample(n=df['sentiment'].value_counts().min(), random_state=42)).reset_index(drop=True)

# Take 5% of the balanced dataset
df_balanced_5_percent = df_balanced.sample(frac=0.05, random_state=42)
# Check the class distribution after taking 5% sample
print(df_balanced_5_percent['sentiment'].value_counts())

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_balanced_5_percent['review_body'], df_balanced_5_percent['sentiment'], test_size=0.2, random_state=42)

# Map string labels to integers AFTER splitting
label_map = {"negative": 0, "positive": 1}
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