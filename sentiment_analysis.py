import torch
from transformers import BertModel
from torch.utils.data import DataLoader, Dataset

from disambiguation import split_text_based_on_conflict


class AlphaCalculator(torch.nn.Module):
    def __init__(self, context_dim, conflict_dim):
        """
        Alpha Calculator dynamically computes alpha based on context and conflict features.
        """
        super(AlphaCalculator, self).__init__()
        self.context_layer = torch.nn.Linear(context_dim, 1)  # Context importance weight
        self.conflict_layer = torch.nn.Linear(conflict_dim, 1)  # Conflict importance weight
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, context_vector, part1_vector, part2_vector):
        """
        Compute alpha dynamically based on context and conflict features.
        """
        conflict_vector = part1_vector - part2_vector
        alpha = self.sigmoid(self.context_layer(context_vector) + self.conflict_layer(conflict_vector))
        return alpha.squeeze()

class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}, self.labels[idx]

    def __len__(self):
        return len(self.labels)

def compute_sentiment_vector(text, sentiment_model, tokenizer):
    """
    Compute the sentiment vector for a given text using the sentiment model.
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = sentiment_model(**inputs)
    return torch.softmax(outputs.logits, dim=1).squeeze()

def compute_context_vector(text, bert_model, tokenizer):
    """
    Compute the context vector for a given text using BERT's [CLS] token output.
    """
    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)  # Use BertModel instead of BertForSequenceClassification
    return outputs.pooler_output.squeeze()  # Extract [CLS] token representation

def fine_tune_model(sentiment_model, tokenizer, train_texts, train_labels, max_length, epochs=10, batch_size=8, save_path="sentiment_model.pt"):
    """
    Train and save a fine-tuned sentiment analysis model using BERT.
    """
    train_texts = list(map(str, train_texts))

    # Tokenize data
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=max_length)
    train_dataset = SentimentDataset(train_encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training setup
    optimizer = torch.optim.AdamW(sentiment_model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    sentiment_model.train()
    for epoch in range(epochs):
        print(f"Current epoch {epoch} out of {epochs}")
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Forward pass
            outputs = sentiment_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

    # Save the fine-tuned model
    torch.save(sentiment_model.state_dict(), save_path)
    print(f"Sentiment model weights saved to {save_path}")

    return sentiment_model, tokenizer


def train_alpha_calculator(
    sentiment_model, train_texts, train_labels, tokenizer, confilict_dim=3, max_length=128, epochs=20, batch_size=8, save_path="alpha_calculator_10epoch_sentiment.pt"
):
    """
    Train the Alpha Calculator using processed texts and save the model weights.
    """
    train_texts = list(map(str, train_texts))

    # Initialize Alpha Calculator
    alpha_calculator = AlphaCalculator(context_dim=768, conflict_dim=confilict_dim)

    # Tokenize data
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=max_length)
    train_dataset = SentimentDataset(train_encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training setup
    optimizer = torch.optim.AdamW(alpha_calculator.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    alpha_calculator.train()
    total_batches = len(train_loader)  # Get total number of batches
    for epoch in range(epochs):
        total_loss = 0
        print(f"Epoch {epoch + 1}/{epochs} started...")
        for batch_idx, batch in enumerate(train_loader):
            inputs, labels = batch
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            batch_mixed_outputs = []  # Store mixed outputs for the batch

            # Step 1: Process each sentence in the batch
            for i in range(len(input_ids)):
                with torch.no_grad():
                    sentence = tokenizer.decode(input_ids[i], skip_special_tokens=True)
                    context_vector = compute_context_vector(sentence, sentiment_model, tokenizer)

                # Step 2: Split sentence into Part 1 and Part 2
                part1_text, part2_text = split_text_based_on_conflict(sentence, tokenizer)

                # Step 3: Generate Part 1 and Part 2 sentiment vectors
                part1_vector = compute_sentiment_vector(part1_text, sentiment_model, tokenizer)
                part2_vector = compute_sentiment_vector(part2_text, sentiment_model, tokenizer) if part2_text else torch.zeros_like(part1_vector)
                print(part1_vector)
                print(part2_vector)

                # Step 4: Compute alpha
                alpha = alpha_calculator(context_vector, part1_vector, part2_vector)

                # Step 5: Compute mixed output for the sentence
                mixed_output = alpha * part1_vector + (1 - alpha) * part2_vector
                batch_mixed_outputs.append(mixed_output)

            # Convert mixed outputs into a batch tensor
            batch_mixed_outputs = torch.stack(batch_mixed_outputs)  # Shape: (batch_size, num_classes)

            # Step 6: Compute loss using hard labels
            loss = criterion(batch_mixed_outputs, labels)
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Print batch progress
            print(f"Batch {batch_idx + 1}/{total_batches}: Loss = {loss.item():.4f}")

        # Print epoch summary
        print(f"Epoch {epoch + 1}/{epochs} completed. Average Loss = {total_loss / total_batches:.4f}")

    # Save the trained AlphaCalculator weights
    torch.save(alpha_calculator.state_dict(), save_path)
    print(f"Alpha Calculator weights saved to {save_path}")

    return alpha_calculator



def evaluate_model(processed_texts, test_labels, sentiment_model, tokenizer, alpha_calculator):
    """
    Evaluate the sentiment model and Alpha Calculator.
    """
    sentiment_model.eval()
    alpha_calculator.eval()
    bert_model = BertModel.from_pretrained('bert-base-uncased')

    predictions = []
    for text in processed_texts:
        # Split text into Part 1 and Part 2
        parts = text.split("[SEP]")
        part1, part2 = parts[0], parts[1] if len(parts) > 1 else ""

        # Compute sentiment vectors
        part1_vector = compute_sentiment_vector(part1, sentiment_model, tokenizer)
        part2_vector = compute_sentiment_vector(part2, sentiment_model, tokenizer) if part2 else torch.zeros_like(part1_vector)

        # Compute context vector for Part 1
        context_vector = compute_context_vector(part1, bert_model, tokenizer)

        # Compute alpha and fused output
        alpha = alpha_calculator(context_vector, part1_vector, part2_vector)
        fused_vector = alpha * part1_vector + (1 - alpha) * part2_vector

        # Predict label
        predictions.append(torch.argmax(fused_vector).item())

    # Evaluate predictions
    accuracy = sum(1 for p, t in zip(predictions, test_labels) if p == t) / len(test_labels)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    return accuracy
