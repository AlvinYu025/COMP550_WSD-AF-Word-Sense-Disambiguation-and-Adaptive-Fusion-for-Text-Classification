import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import opinion_lexicon, wordnet
import spacy
import nltk

nltk.download('opinion_lexicon')
nltk.download('wordnet')
nltk.download('punkt')

# Load opinion lexicons
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())

# Load spaCy model for dependency parsing
nlp = spacy.load("en_core_web_sm")


def extract_dominant_phrases(sentence):
    """
    Extract dominant phrases (subject, predicate, object) using spaCy.
    Args:
        sentence (str): Input sentence.
    Returns:
        dict: Dominant phrases with keys 'subject', 'predicate', 'object'.
    """
    if not isinstance(sentence, str):
        sentence = str(sentence)  # Ensure the sentence is a string

    doc = nlp(sentence)
    dominant_phrases = {"subject": None, "predicate": None, "object": None}

    for token in doc:
        if token.dep_ == "nsubj":  # Subject
            dominant_phrases["subject"] = token.text
        elif token.dep_ == "ROOT":  # Predicate
            dominant_phrases["predicate"] = token.text
        elif token.dep_ in {"dobj", "pobj"}:  # Object
            dominant_phrases["object"] = token.text
        elif token.dep_ == "amod":  # Adjective modifying a noun
            dominant_phrases["adjective"] = token.text

    return dominant_phrases


def disambiguate_word(word, context):
    """
    Disambiguate the meaning of a word in the given context using a pre-trained WSD model.
    Args:
        word (str): Target word to disambiguate.
        context (str): Context in which the word appears.
    Returns:
        str: Word with its disambiguated meaning (or the original word if no definition is found).
    """
    # Use WordNet for candidate senses
    synsets = wordnet.synsets(word)
    if not synsets:
        return word  # If no synsets are found, return the original word

    # Pre-trained WSD model (e.g., from Hugging Face)
    # Load model directly

    tokenizer = AutoTokenizer.from_pretrained("MiMe-MeMo/MeMo-BERT-WSD")
    model = AutoModelForSequenceClassification.from_pretrained("MiMe-MeMo/MeMo-BERT-WSD")

    # Format input for the WSD model
    inputs = tokenizer(f"{context} [SEP] {word}", return_tensors="pt", truncation=True, padding=True)

    # Predict sense
    with torch.no_grad():
        outputs = model(**inputs)
    sense_index = torch.argmax(outputs.logits).item()

    # Map sense index to WordNet synset
    if sense_index < len(synsets):
        sense_definition = synsets[sense_index].definition()
        return f"{word} ({sense_definition})"
    else:
        return word


def process_sentences(sentences, labels, model, tokenizer):
    """
    Process sentences by extracting dominant phrases and dynamically calculating weighted sentiment.
    Args:
        sentences (list): List of input sentences.
        labels (list): List of corresponding labels for the sentences.
        model: Sentiment analysis model used for computing sentiment vectors.
        tokenizer: Tokenizer used with the sentiment model.
    Returns:
        list: Processed sentences with disambiguated dominant phrases, sentiment vectors, and processed sentence text.
    """
    processed_sentences = []

    for sentence, label in zip(sentences, labels):  # Ensure sentence and label are processed together
        sentence = str(sentence)  # Ensure the sentence is a string

        # Extract dominant phrases
        dominant_phrases = extract_dominant_phrases(sentence)
        subject = dominant_phrases.get("subject")
        predicate = dominant_phrases.get("predicate")
        obj = dominant_phrases.get("object")
        adjective = dominant_phrases.get("adjective")  # Extract adjectives if available

        # Disambiguate extracted phrases
        subject_disambiguated = disambiguate_word(subject, sentence) if subject else None
        predicate_disambiguated = disambiguate_word(predicate, sentence) if predicate else None
        obj_disambiguated = disambiguate_word(obj, sentence) if obj else None
        adjective_disambiguated = disambiguate_word(adjective, sentence) if adjective else None

        # Concatenate disambiguated phrases for focused sentiment computation
        dominant_context = " ".join(filter(None, [subject, predicate, obj, adjective])).strip()
        wsd_dominant_context = " ".join(filter(None, [
            subject_disambiguated, predicate_disambiguated, obj_disambiguated
        ])).strip()

        # Form the processed sentence
        processed_sentence = sentence
        if subject_disambiguated:
            processed_sentence = processed_sentence.replace(subject, subject_disambiguated, 1)
        if predicate_disambiguated:
            processed_sentence = processed_sentence.replace(predicate, predicate_disambiguated, 1)
        if obj_disambiguated:
            processed_sentence = processed_sentence.replace(obj, obj_disambiguated, 1)
        if adjective_disambiguated:
            processed_sentence = processed_sentence.replace(adjective, adjective_disambiguated, 1)

        # Compute sentiment vectors
        global_vector = compute_global_sentiment(sentence, model, tokenizer)
        context_vector = compute_global_sentiment(dominant_context, model, tokenizer)

        wsd_global_vector = compute_global_sentiment(processed_sentence, model, tokenizer)
        wsd_context_vector = compute_global_sentiment(wsd_dominant_context, model, tokenizer)

        # Append processed sentence data
        processed_sentences.append({
            "text": sentence,
            "label": label,
            "processed_sentence": processed_sentence,
            "global_vector": global_vector,
            "context_vector": context_vector,
            "wsd_global_vector": wsd_global_vector,
            "wsd_context_vector": wsd_context_vector,
            "disambiguated_phrases": {
                "subject": subject_disambiguated,
                "predicate": predicate_disambiguated,
                "object": obj_disambiguated,
                "adjective": adjective_disambiguated
            }
        })

    return processed_sentences


def compute_global_sentiment(sentence, model, tokenizer):
    """
    Compute the global sentiment vector for a given sentence.
    Args:
        sentence (str): The input sentence.
        model: The sentiment model.
        tokenizer: The tokenizer for the model.
    Returns:
        torch.Tensor: The sentiment vector.
    """
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.softmax(outputs.logits, dim=1).squeeze()

