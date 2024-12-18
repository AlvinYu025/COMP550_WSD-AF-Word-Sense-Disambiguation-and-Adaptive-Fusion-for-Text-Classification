import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torch.utils.data import Dataset, DataLoader

class AlphaCalculator(torch.nn.Module):
    def __init__(self, vector_dim):
        super(AlphaCalculator, self).__init__()
        self.weight_net = torch.nn.Sequential(
            torch.nn.Linear(vector_dim * 2, 64),  # Combine global and context
            torch.nn.ReLU(),
            torch.nn.Linear(64, vector_dim * 2),  # Output weights for each dimension
            torch.nn.Softmax(dim=-1)  # Normalize weights across dimensions
        )

    def forward(self, global_vector, context_vector):
        # Concatenate vectors
        combined_input = torch.cat([global_vector, context_vector], dim=-1)

        # Compute weights
        weights = self.weight_net(combined_input)  # Shape: [batch_size, vector_dim * 2]
        w_global, w_context = weights[:, :global_vector.size(1)], weights[:, global_vector.size(1):]

        # Compute fused vector
        fused_vector = w_global * global_vector + w_context * context_vector
        return fused_vector


def fine_tune_model(sentiment_model, tokenizer, train_texts, train_labels, val_texts, val_labels,
                    max_length, epochs=10, batch_size=8, save_path="FT_RoBERTa_10epoch.pt"):
    """
    Train and save a fine-tuned sentiment analysis model using BERT/RoBERTa.
    Includes validation after each epoch to monitor performance.
    """
    # Convert train and val texts to strings
    train_texts = list(map(str, train_texts))
    val_texts = list(map(str, val_texts))

    # Prepare training data
    train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=max_length)
    train_dataset = SentimentDataset(train_encodings, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Prepare validation data
    val_encodings = tokenizer(val_texts, truncation=True, padding=True, max_length=max_length)
    val_dataset = SentimentDataset(val_encodings, val_labels)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define optimizer and loss criterion
    optimizer = torch.optim.AdamW(sentiment_model.parameters(), lr=2e-5)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")  # Track the best validation loss
    best_model_path = save_path

    sentiment_model.train()
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        total_loss = 0

        # Training loop
        for batch in train_loader:
            inputs, labels = batch
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]

            outputs = sentiment_model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Training Loss: {total_loss / len(train_loader):.4f}")

        # Validation loop
        sentiment_model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                outputs = sentiment_model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(sentiment_model.state_dict(), best_model_path)
            print(f"New best model saved to {best_model_path}")

        sentiment_model.train()

    print(f"Training completed. Best model saved at {best_model_path} with Validation Loss: {best_val_loss:.4f}")
    return sentiment_model


def train_alpha_calculator(
    train_data, val_data, epochs=100, batch_size=8, save_path="best_alpha_calculator_weights.pt"
):
    """
    Train the Alpha Calculator using processed data with validation.
    Args:
        train_data (list): Processed training sentences with global and context vectors.
        val_data (list): Processed validation sentences with global and context vectors.
        epochs (int): Number of training epochs.
        batch_size (int): Batch size for training.
        save_path (str): Path to save the best-performing model weights.

    Returns:
        AlphaCalculator: The trained Alpha Calculator model.
    """
    vector_dim = 3  # Sentiment vector dimension
    alpha_calculator = AlphaCalculator(vector_dim=vector_dim)

    optimizer = torch.optim.AdamW(alpha_calculator.parameters(), lr=2e-4)
    criterion = torch.nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    def process_batch(batch):
        """
        Helper function to process a batch of data into tensors.
        """
        global_vectors = torch.stack([item["global_vector"] for item in batch])
        context_vectors = torch.stack([item["context_vector"] for item in batch])
        labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
        return global_vectors, context_vectors, labels

    for epoch in range(epochs):
        total_train_loss = 0
        print(f"Epoch {epoch + 1}/{epochs}")

        # Training Loop
        alpha_calculator.train()
        for i in range(0, len(train_data), batch_size):
            batch = train_data[i:i + batch_size]

            global_vectors, context_vectors, labels = process_batch(batch)

            fused_vectors = alpha_calculator(global_vectors, context_vectors)
            loss = criterion(fused_vectors, labels)
            total_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_train_loss = total_train_loss / (len(train_data) // batch_size)
        print(f"Training Loss: {avg_train_loss:.4f}")

        # Validation Loop
        alpha_calculator.eval()
        total_val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for i in range(0, len(val_data), batch_size):
                batch = val_data[i:i + batch_size]

                global_vectors, context_vectors, labels = process_batch(batch)

                fused_vectors = alpha_calculator(global_vectors, context_vectors)
                loss = criterion(fused_vectors, labels)
                total_val_loss += loss.item()

                # Collect predictions and true labels for accuracy calculation
                preds = torch.argmax(fused_vectors, dim=1).tolist()
                all_preds.extend(preds)
                all_labels.extend(labels.tolist())

        avg_val_loss = total_val_loss / (len(val_data) // batch_size)
        val_acc = accuracy_score(all_labels, all_preds)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_acc * 100:.2f}%")

        # Save the best model weights
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(alpha_calculator.state_dict(), save_path)
            print(f"New best model weights saved to {save_path}")

    print(f"Training completed. Best Validation Loss: {best_val_loss:.4f}")
    return alpha_calculator


def evaluate_model(mode, test_texts, test_labels, sentiment_model, alpha_calculator=None):
    """
    Dynamically evaluate the model based on the presence of an Alpha Calculator.
    Compute multiple evaluation metrics including Accuracy, Precision, Recall, and F1-Score.

    Args:
        test_texts (list): List of test sentences.
        test_labels (list): List of true sentiment labels for the test set.
        sentiment_model: The fine-tuned BERT/RoBERTa sentiment model.
        tokenizer: Tokenizer used for the sentiment model.
        alpha_calculator (optional): Alpha Calculator model for fusing sentiment vectors.

    Returns:
        dict: Dictionary containing evaluation metrics (Accuracy, Precision, Recall, F1-Score).
    """
    sentiment_model.eval()  # Set the sentiment model to evaluation mode
    if alpha_calculator:
        alpha_calculator.eval()  # Set the Alpha Calculator to evaluation mode

    predictions = []

    with torch.no_grad():
        for data in test_texts:
            # Step 1: Compute the global vector
            if mode == "Fusion" or mode == "baseline":
                global_vector = data["global_vector"]
            else:
                global_vector = data["wsd_global_vector"]

            if alpha_calculator:
                # Step 2: Extract context vector via process_sentences
                if mode == "baseline":
                    context_vector = data["context_vector"]
                else:
                    context_vector = data["wsd_context_vector"]

                # Step 3: Compute fused output using Alpha Calculator
                fused_vector = alpha_calculator(global_vector, context_vector)
                predicted_label = torch.argmax(fused_vector).item()
            else:
                # Directly predict using the global vector from BERT/RoBERTa
                predicted_label = torch.argmax(global_vector).item()

            predictions.append(predicted_label)

    # Step 4: Compute metrics
    acc = accuracy_score(test_labels, predictions)
    precision = precision_score(test_labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(test_labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(test_labels, predictions, average='weighted', zero_division=0)

    # Print metrics
    print(f"Accuracy: {acc * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-Score: {f1 * 100:.2f}%")

    # Return metrics as a dictionary
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1_score": f1
    }


class SentimentDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}, self.labels[idx]

    def __len__(self):
        return len(self.labels)
