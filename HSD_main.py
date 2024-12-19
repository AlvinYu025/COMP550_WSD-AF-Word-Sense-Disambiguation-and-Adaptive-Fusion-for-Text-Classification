import pandas as pd
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from disambiguation import process_sentences
from sentiment_analysis import FusionModel, fine_tune_model, evaluate_model, train_FusionModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def load_Smodel_dataset(data_path=None):
    """
    Load Sentiment Model dataset from a single CSV file.
    The dataset must contain at least two columns: 'Feedback' and 'Sentiment Label'.
    Args:
        data_path (str): Path to the dataset CSV file.
    Returns:
        train_df (DataFrame): Training data.
        test_df (DataFrame): Test data.
    """
    if not data_path:
        raise ValueError("A valid data path must be provided for Sentiment Model dataset.")

    # Load the dataset
    df = pd.read_csv(data_path, encoding='latin1', header=None)
    df.columns = ['Feedback', 'Sentiment Label', 'Ratings', 'Extra']  # Rename columns

    # Drop the extra column if it exists
    df = df.drop(columns=['Extra'], errors='ignore')

    # Ensure proper data types
    df['Feedback'] = df['Feedback'].astype(str)
    df['Sentiment Label'] = pd.to_numeric(df['Sentiment Label'], errors='coerce')  # Convert safely to numeric
    df = df.dropna(subset=['Sentiment Label'])  # Drop invalid rows
    df['Sentiment Label'] = df['Sentiment Label'].astype(int)

    # Split into training and testing datasets
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42
    )

    return train_df, test_df


def load_fusion_dataset(train_path, test_path):
    """
    Load Alpha Calculator dataset from TXT files.
    Each line in the TXT file should be formatted as 'sentence,label'.
    """
    def load_txt(file_path):
        sentences, labels = [], []
        with open(file_path, "r", encoding="utf-8") as file:
            for line in file:
                parts = line.strip().split(",")
                if len(parts) == 2:
                    sentence, label = parts
                    sentences.append(sentence)
                    labels.append(int(label))  # Ensure labels are integers
        return sentences, labels

    train_sentences, train_labels = load_txt(train_path)
    test_sentences, test_labels = load_txt(test_path)

    return train_sentences, train_labels, test_sentences, test_labels


def preprocess_labels(df):
    """
    Map sentiment labels to integers for Sentiment Model datasets.
    """
    label_mapping = {0: 1, 1: 0}
    df['Sentiment Label'] = df['Sentiment Label'].map(label_mapping)
    return df


def evaluate_mode(mode, test_texts, test_labels, sentiment_model, tokenizer, alpha_calculator=None):
    """
    Evaluate the model dynamically based on the selected mode.
    """
    processed_test = process_sentences(test_texts, test_labels, sentiment_model, tokenizer)
    if mode == "baseline":
        return evaluate_model(mode, processed_test, test_labels, sentiment_model)

    elif mode == "WSD":
        return evaluate_model(mode, processed_test, test_labels, sentiment_model)

    elif mode == "AF":
        return evaluate_model(mode, processed_test, test_labels, sentiment_model, alpha_calculator)

    elif mode == "WSD_AF":
        return evaluate_model(mode, processed_test, test_labels, sentiment_model, alpha_calculator)

    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    # Configuration
    DATA_PATH = 'dataset/hospital.csv'
    fusion_TRAIN_PATH = 'dataset/fusionModel/trainset.txt'
    fusion_TEST_PATH = 'dataset/fusionModel/testset.txt'
    dataset_name = 'Hospital Review Dataset'
    TEST_SIZE = 0.2
    VALIDATION_SPLIT = 0.2  # Fraction of data for validation
    RANDOM_STATE = 42
    MAX_LENGTH = 128
    mode = "WSD"  # Options: baseline, WSD, AF, WSD_AF

    model_name = "bert-base-uncased"
    # model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 1: Load Sentiment Model dataset
    print(f"Step 1: Load Sentiment Model dataset {dataset_name}")
    if DATA_PATH:
        train_df, test_df = load_Smodel_dataset(data_path=DATA_PATH)

    train_df = preprocess_labels(train_df)
    test_df = preprocess_labels(test_df)

    # Split training set into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['Feedback'].tolist(), train_df['Sentiment Label'].tolist(),
        test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
    )
    test_texts = test_df['Feedback'].tolist()
    test_labels = test_df['Sentiment Label'].tolist()

    print(f"Number of training examples: {len(train_texts)}")
    print(f"Number of validation examples: {len(val_texts)}")
    print(f"Number of test examples: {len(test_texts)}")

    # Step 2: Fine-tune the pre-trained model if not in baseline mode
    print("Step 2: Fine-tune the pre-trained model")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    # fine_tuned_model_path = "FT_HRD_BERT_10epoch.pt"
    # fine_tuned_model_path = "FT_HRD_RoBERTa_10epoch.pt"
    # sentiment_model.load_state_dict(torch.load(fine_tuned_model_path))
    sentiment_model = fine_tune_model(
        sentiment_model, tokenizer, train_texts, train_labels, val_texts, val_labels,
        max_length=MAX_LENGTH, epochs=10, batch_size=32, save_path="FT_HRD_RoBERTa_10epoch.pt"
    )
    sentiment_model.eval()

    # Step 3: Load Fusion dataset
    print("Step 3: Load Fusion dataset")
    train_texts_fusion, train_labels_fusion, test_texts_fusion, test_labels_fusion = load_fusion_dataset(
        fusion_TRAIN_PATH, fusion_TEST_PATH
    )

    # Step 4: Process Fusion data
    print("Step 4: Process Fusion data")
    processed_train_fusion = process_sentences(train_texts_fusion, train_labels_fusion, sentiment_model, tokenizer)
    processed_test_fusion = process_sentences(test_texts_fusion, test_labels_fusion, sentiment_model, tokenizer)

    # Combine Sentiment Model training data and Fusion training data
    print("Step 5: Combine training data")
    full_train_data = processed_train_fusion + process_sentences(train_texts, train_labels, sentiment_model, tokenizer)
    full_test_data = processed_test_fusion + process_sentences(test_texts, test_labels, sentiment_model, tokenizer)

    # Step 6: Train Fusion Model if required
    fusionModel = None

    fusionModel = FusionModel(vector_dim=2)
    # fusion_model_path = "BERT_HRD_AF.pt"
    # # fusion_model_path = "RoBERTa_HRD_AF.pt"
    # fusionModel.load_state_dict(torch.load(fusion_model_path))

    print("Step 6: Train Fusion Model")
    fusionModel = train_FusionModel(
        full_train_data, full_test_data, epochs=200, batch_size=8, save_path="BERT_HRD_AF.pt"
    )

    # Step 7: Evaluate the model
    print("Step 7: Evaluate the model")
    evaluate_mode(mode, test_texts+test_texts_fusion, test_labels+test_labels_fusion, sentiment_model, tokenizer, fusionModel)
