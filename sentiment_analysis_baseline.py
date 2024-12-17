import numpy as np
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from torch.nn.functional import softmax
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Prepare dataset for PyTorch
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # Include the labels directly in the returned dictionary
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])  # Add labels here
        return item

    def __len__(self):
        return len(self.labels)

# Train and evaluate the sentiment analysis model
def train_and_evaluate_model(train_texts, train_labels, test_texts, test_labels, max_length):
    # Load tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

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