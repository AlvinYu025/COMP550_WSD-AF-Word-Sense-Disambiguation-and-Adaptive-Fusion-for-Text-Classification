import pandas as pd
from sklearn.model_selection import train_test_split
from entity_recognition import perform_ner
from disambiguation import disambiguate
from sentiment_analysis import train_and_evaluate_model

# Set parameters
DATA_PATH = "dataset/sentiment140.csv"
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_LENGTH = 128

# Load and preprocess the dataset
def load_dataset(path):
    df = pd.read_csv(path, encoding='latin1', header=None)
    df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']
    df['sentiment'] = df['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
    df = df[['text', 'sentiment']]
    return df

if __name__ == "__main__":
    # Load dataset
    df = load_dataset(DATA_PATH)

    # Split dataset into training and test sets
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        df['text'], df['sentiment'], test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    # Perform Named Entity Recognition (NER) to extract target words
    target_data = perform_ner(train_texts)

    # Perform Word Sense Disambiguation (WSD) on target words
    modified_texts = disambiguate(train_texts, target_data)

    # Train and evaluate sentiment analysis model
    train_and_evaluate_model(modified_texts, train_labels, test_texts, test_labels, MAX_LENGTH)
