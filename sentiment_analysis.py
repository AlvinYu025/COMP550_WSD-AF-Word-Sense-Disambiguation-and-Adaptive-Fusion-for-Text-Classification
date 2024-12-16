import torch
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
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
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.pooler_output.squeeze()

def fine_tune_model(sentiment_model, tokenizer, train_texts, train_labels, max_length, epochs=10, batch_size=8):
    """
    Train a sentiment analysis model using BERT.
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

    return sentiment_model, tokenizer

def train_alpha_calculator(sentiment_model, train_texts, train_labels, tokenizer, confilict_dim=3, max_length=3, epochs=20, batch_size=8):
    """
    Train the Alpha Calculator using processed texts.
    """
    alpha_calculator = AlphaCalculator(context_dim=768, conflict_dim=confilict_dim)

    # Tokenize data
    train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=max_length)
    train_dataset = SentimentDataset(train_encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Training setup
    optimizer = torch.optim.AdamW(alpha_calculator.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    alpha_calculator.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            inputs, labels = batch
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            # Step 1: Extract context vector
            with torch.no_grad():
                context_vector = compute_context_vector(input_ids, sentiment_model, tokenizer)

            # Step 2: Split sentence into Part 1 and Part 2
            part1_text, part2_text = split_text_based_on_conflict(input_ids, tokenizer)

            # Step 3: Generate Part 1 and Part 2 sentiment vectors
            part1_vector = compute_sentiment_vector(part1_text, sentiment_model, tokenizer)
            part2_vector = compute_sentiment_vector(part2_text, sentiment_model, tokenizer) if part2_text else torch.zeros_like(part1_vector)

            # Step 4: Compute alpha
            alpha = alpha_calculator(context_vector, part1_vector, part2_vector)

            # Step 5: Compute mixed output
            mixed_output = alpha * part1_vector + (1 - alpha) * part2_vector

            # Step 6: Compute loss using hard labels
            loss = criterion(mixed_output.unsqueeze(0), labels)
            total_loss += loss.item()

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

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
