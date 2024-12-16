import pandas as pd
from sklearn.model_selection import train_test_split

from target_word_recognition import perform_ner
from disambiguation import process_sentences
from sentiment_analysis import fine_tune_model, evaluate_model, train_alpha_calculator
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_dataset(data_path=None, train_path=None, test_path=None):
    if train_path and test_path:
        train_df = pd.read_csv(train_path, encoding='latin1')
        test_df = pd.read_csv(test_path, encoding='latin1')
    elif data_path:
        df = pd.read_csv(data_path)
        train_df, test_df = train_test_split(
            df, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
    else:
        raise ValueError("No valid dataset path provided.")

    return train_df, test_df


def preprocess_labels(df, dataset_name):
    if dataset_name.lower() == "sentiment_analysis_dataset":
        # Change labels
        label_mapping = {'positive': 0, 'neutral': 1, 'negative': 2}
        df['sentiment'] = df['sentiment'].map(label_mapping)
    # other datasets
    return df


if __name__ == "__main__":
    DATA_PATH = None
    TRAIN_PATH = 'dataset/Sentiment_Analysis_Dataset/train.csv'
    TEST_PATH = 'dataset/Sentiment_Analysis_Dataset/test.csv'
    dataset_name = 'Sentiment_Analysis_Dataset'
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_LENGTH = 128
    model_name = "bert-base-uncased"  # Example: BERT model
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if TRAIN_PATH and TEST_PATH:
        train_df, test_df = load_dataset(train_path=TRAIN_PATH, test_path=TEST_PATH)
    elif DATA_PATH:
        train_df, test_df = load_dataset(data_path=DATA_PATH)
    else:
        raise ValueError("No dataset path provided.")

    train_df = preprocess_labels(train_df, dataset_name)
    test_df = preprocess_labels(test_df, dataset_name)

    if dataset_name.lower() == "sentiment_analysis_dataset":
        train_texts = train_df['text']
        train_labels = train_df['sentiment']
        test_texts = test_df['text']
        test_labels = test_df['sentiment']

    # Optionally, print some information about the datasets
    print(f"Number of training examples: {len(train_texts)}")
    print(f"Number of test examples: {len(test_texts)}")
    print(f"Example train text: [{train_texts.iloc[0]}] with sentiment label: {train_labels.iloc[0]}")

    # exit()

    # Step 2: Fine-tune the pre-trained model
    sentiment_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    sentiment_model = fine_tune_model(sentiment_model, tokenizer, train_texts, train_labels, max_length=MAX_LENGTH, epochs=10, batch_size=32)

    # Step 3: Train Alpha Calculator
    alpha_calculator = train_alpha_calculator(sentiment_model, train_texts, train_labels, tokenizer, confilict_dim=3, max_length=MAX_LENGTH, epochs=20, batch_size=32)

    # Step 4: Ambiguous Target Word Recognition
    target_data = perform_ner(train_texts)

    # Step 5: Split sentences based on conflicts and disambiguate words
    processed_texts = process_sentences(target_data)

    # Step 6: Evaluate the model
    evaluate_model(
        processed_texts, test_labels, sentiment_model, tokenizer, alpha_calculator
    )
