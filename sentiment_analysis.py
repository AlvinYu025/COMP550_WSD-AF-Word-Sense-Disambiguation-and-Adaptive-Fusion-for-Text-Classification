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

    # Tokenize texts
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=max_length)
    test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=max_length)

    # Prepare datasets
    train_dataset = SentimentDataset(train_encodings, train_labels)
    test_dataset = SentimentDataset(test_encodings, test_labels)

    # Training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=4,
        per_device_train_batch_size=12,
        per_device_eval_batch_size=12,
        eval_strategy="epoch",
        save_total_limit=2,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Test sample predictions
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

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    
    # Get predicted class indices (same as before)
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
        'confidence_scores': confidence_scores.tolist()  # Returning the confidence scores
    }