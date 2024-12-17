import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from torch.nn.functional import softmax

# Load the dataset
df = pd.read_csv('dataset/sentiment140.csv', encoding='latin1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Map sentiment labels
df['sentiment'] = df['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
df = df[['text', 'sentiment']]  # Retain only text and sentiment columns

# Sample 10% of data for each sentiment class
df_sampled = df.groupby('sentiment', group_keys=False).apply(lambda x: x.sample(frac=0.001, random_state=42)).reset_index(drop=True)

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df_sampled['text'], df_sampled['sentiment'], test_size=0.2, random_state=42)

# Map string labels to integers AFTER splitting
label_map = {"negative": 0, "neutral": 1, "positive": 2}
train_labels = [label_map[label] for label in train_labels]
test_labels = [label_map[label] for label in test_labels]

from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments

# Load tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForSequenceClassification.from_pretrained('roberta-base', num_labels=3)
'''''
# Load the fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained('./final_model')
tokenizer = BertTokenizer.from_pretrained('./final_model')
'''
# Tokenize the dataset

train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)


# Prepare PyTorch datasets
import torch
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


# Verify lengths match
assert len(train_encodings['input_ids']) == len(train_labels), "Mismatch in lengths of encodings and labels"
assert len(test_encodings['input_ids']) == len(test_labels), "Mismatch in lengths of encodings and labels"

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# Train the model
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=2,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=3e-5,
    weight_decay=0.01,
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Get predicted class indices
    preds = np.argmax(logits, axis=-1)
    
    # Calculate accuracy
    accuracy = accuracy_score(labels, preds)
    
    # Calculate precision, recall, and F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    
    # Convert logits to probabilities using softmax (for confidence score)
    probs = softmax(torch.tensor(logits), dim=-1)
    
    # Get the confidence scores (probability of the predicted class)
    confidence_scores = probs.max(dim=-1).values.numpy()  # Max probability corresponds to predicted class
    
    # Optionally: Return the confidence scores for inspection, but generally metrics are returned
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        #'confidence_scores': confidence_scores.tolist()  # Returning the confidence scores
    }

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset, compute_metrics=compute_metrics,
)

trainer.train()
# Save the fine-tuned model and tokenizer
model.save_pretrained('./final_RoBERTa_model')
tokenizer.save_pretrained('./final_RoBERTa_model')

############ TESTING #############

# Evaluate on the test set
trainer.evaluate()

inputs = tokenizer(test_texts.tolist(), truncation=True, padding=True, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1).tolist()

# Calculate Accuracy and F1 Scores
true_labels = test_labels
accuracy = accuracy_score(true_labels, predictions)
f1 = f1_score(true_labels, predictions, average='weighted')  # Use 'weighted' for multi-class

# Print the metrics
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")