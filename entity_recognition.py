from transformers import pipeline

# Perform Named Entity Recognition (NER) and return the input sentence with target words
def perform_ner(texts):
    # Load pre-trained NER model
    ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")

    # Perform NER on texts
    target_data = []
    for text in texts:
        entities = ner_pipeline(text)
        target_data.append({"text": text, "entities": [e['word'] for e in entities if 'word' in e]})
    return target_data
