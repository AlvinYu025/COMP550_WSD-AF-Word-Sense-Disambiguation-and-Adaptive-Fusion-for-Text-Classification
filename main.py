import pandas as pd
import random
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from disambiguation import process_sentences
from sentiment_analysis import fine_tune_model, evaluate_model, train_alpha_calculator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

def load_Smodel_dataset(data_path=None, train_path=None, test_path=None):
    """
    Load Sentiment Model dataset from CSV files for fine-tuning BERT.
    """
    if train_path and test_path:
        train_df = pd.read_csv(train_path, encoding='latin1').head(2000)
        test_df = pd.read_csv(test_path, encoding='latin1').head(400)
    elif data_path:
        df = pd.read_csv(data_path)
        train_df, test_df = train_test_split(
            df, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    else:
        raise ValueError("No valid dataset path provided for Sentiment Model.")

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


def preprocess_labels(df, dataset_name):
    """
    Map sentiment labels to integers for Sentiment Model datasets.
    """
    if dataset_name.lower() == "sentiment_analysis_dataset":
        label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
        df['sentiment'] = df['sentiment'].map(label_mapping)
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

    elif mode == "Fusion":
        return evaluate_model(mode, processed_test, test_labels, sentiment_model, alpha_calculator)

    elif mode == "WSD_Fusion":
        return evaluate_model(mode, processed_test, test_labels, sentiment_model, alpha_calculator)

    else:
        raise ValueError(f"Invalid mode: {mode}")


if __name__ == "__main__":
    # Configuration
    DATA_PATH = None
    Smodel_TRAIN_PATH = 'dataset/Sentiment_Analysis_Dataset/train.csv'
    Smodel_TEST_PATH = 'dataset/Sentiment_Analysis_Dataset/test.csv'
    fusion_TRAIN_PATH = 'dataset/alpha_calculator/trainset.txt'
    fusion_TEST_PATH = 'dataset/alpha_calculator/testset.txt'
    dataset_name = 'Sentiment_Analysis_Dataset'
    TEST_SIZE = 0.2
    VALIDATION_SPLIT = 0.2  # Fraction of data for validation
    RANDOM_STATE = 42
    MAX_LENGTH = 128
    mode = "baseline"  # Options: baseline, WSD, Fusion, WSD_Fusion

    model_name = "bert-base-uncased"
    # model_name = "roberta-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Step 1: Load Sentiment Model dataset
    print(f"Step 1: Load Sentiment Model dataset {dataset_name}")
    if Smodel_TRAIN_PATH and Smodel_TEST_PATH:
        train_df, test_df = load_Smodel_dataset(train_path=Smodel_TRAIN_PATH, test_path=Smodel_TEST_PATH)
    elif DATA_PATH:
        train_df, test_df = load_Smodel_dataset(data_path=DATA_PATH)
    else:
        raise ValueError("No dataset path provided for Sentiment Model.")

    train_df = preprocess_labels(train_df, dataset_name)
    test_df = preprocess_labels(test_df, dataset_name)

    # Split training set into training and validation sets
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        train_df['text'].tolist(), train_df['sentiment'].tolist(),
        test_size=VALIDATION_SPLIT, random_state=RANDOM_STATE
    )
    test_texts = test_df['text'].tolist()
    test_labels = test_df['sentiment'].tolist()

    print(f"Number of training examples: {len(train_texts)}")
    print(f"Number of validation examples: {len(val_texts)}")
    print(f"Number of test examples: {len(test_texts)}")

    # Step 2: Fine-tune the pre-trained model if not in baseline mode
    print("Step 2: Fine-tune the pre-trained model")
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    # fine_tuned_model_path = "FT_SAD_BERT_10epoch.pt"
    # sentiment_model.load_state_dict(torch.load(fine_tuned_model_path))
    # sentiment_model = fine_tune_model(
    #     sentiment_model, tokenizer, train_texts, train_labels, val_texts, val_labels,
    #     max_length=MAX_LENGTH, epochs=10, batch_size=32
    # )
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

    # Step 6: Train Alpha Calculator if required
    alpha_calculator = None
    if mode in {"Fusion", "WSD_Fusion"}:
        print("Step 6: Train Alpha Calculator")
        alpha_calculator = train_alpha_calculator(
            full_train_data, full_test_data, epochs=500, batch_size=8
        )

    # Step 7: Evaluate the model
    print("Step 7: Evaluate the model")
    evaluate_mode(mode, test_texts, test_labels, sentiment_model, tokenizer, alpha_calculator)

