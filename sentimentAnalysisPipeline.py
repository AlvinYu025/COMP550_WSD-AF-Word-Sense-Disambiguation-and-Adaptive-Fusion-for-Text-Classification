'''''
import kagglehub

# Download latest version
path = kagglehub.dataset_download("kazanova/sentiment140")

print("Path to dataset files:", path)
'''
import pandas as pd

# Load the dataset
df = pd.read_csv('dataset/sentiment140.csv', encoding='latin1', header=None)
df.columns = ['target', 'id', 'date', 'flag', 'user', 'text']

# Map sentiment labels
df['sentiment'] = df['target'].map({0: 'negative', 2: 'neutral', 4: 'positive'})
df = df[['text', 'sentiment']]  # Retain only text and sentiment columns

# Split the dataset into training and test sets
from sklearn.model_selection import train_test_split
train_texts, test_texts, train_labels, test_labels = train_test_split(
    df['text'], df['sentiment'], test_size=0.2, random_state=42)


################### DANLIN ##########################

from transformers import pipeline

# FINE TUNE WITH DIFFERENT MODELS suitable NER dataset, such as CoNLL-2003 or OntoNotes
# Load a pre-trained NER model
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

# Perform NER on the tweets
entities = [ner_pipeline(text) for text in train_texts]

#################### FENGFEI ###############

# Example pseudocode for entity linking
from wikidata.client import Client

client = Client()
linked_entities = []
for entity_list in entities:
    linked_entities.append([
        client.get(entity['word']) for entity in entity_list if 'word' in entity
    ])

################

# Example of fetching additional data from Wikidata
for linked_entity in linked_entities:
    for entity in linked_entity:
        if entity:
            # Fetch properties or relationships
            print(entity.attributes)  # Example of retrieving additional context

# NEW DATASET (original dataset + enrichment from linked entities) --> make sure the output is in this structure because my part uses this for input
enriched_texts = []
for original_text, linked_entity in zip(train_texts, linked_entities):
    context = " ".join([entity.attributes.get('description', '') for entity in linked_entity if entity])
    enriched_texts.append(f"{original_text} {context}")


########## MANDY ############

from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments

# Load tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)

# Tokenize the dataset
'''''
train_encodings = tokenizer(list(train_texts), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(test_texts), truncation=True, padding=True, max_length=128)
'''
train_encodings = tokenizer(enriched_texts, truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(test_texts, truncation=True, padding=True, max_length=128)


# Prepare PyTorch datasets
import torch
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}, self.labels[idx]

    def __len__(self):
        return len(self.labels)

train_dataset = SentimentDataset(train_encodings, train_labels)
test_dataset = SentimentDataset(test_encodings, test_labels)

# Train the model
training_args = TrainingArguments(
    output_dir='./results', num_train_epochs=3, per_device_train_batch_size=8,
    per_device_eval_batch_size=8, evaluation_strategy="epoch", save_steps=10_000,
    save_total_limit=2, learning_rate=2e-5, weight_decay=0.01, logging_dir='./logs',
)

trainer = Trainer(
    model=model, args=training_args, train_dataset=train_dataset, eval_dataset=test_dataset
)

trainer.train()

############ TESTING #############

# Evaluate on the test set
trainer.evaluate()

# Predict sentiment for new tweets
test_texts_sample = test_texts[:10].tolist()
inputs = tokenizer(test_texts_sample, truncation=True, padding=True, return_tensors="pt")
outputs = model(**inputs)
predictions = torch.argmax(outputs.logits, dim=-1)
print(predictions)
