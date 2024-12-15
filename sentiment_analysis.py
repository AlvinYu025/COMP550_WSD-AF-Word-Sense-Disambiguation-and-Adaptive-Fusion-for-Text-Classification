from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Prepare dataset for PyTorch
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}, self.labels[idx]

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
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        evaluation_strategy="epoch",
        save_steps=10_000,
        save_total_limit=2,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_dir='./logs',
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    trainer.evaluate()

    # Test sample predictions
    test_texts_sample = test_texts[:10].tolist()
    inputs = tokenizer(test_texts_sample, truncation=True, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=-1)
    print("Predictions:", predictions)
