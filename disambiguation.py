import torch
import nltk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from nltk.corpus import opinion_lexicon, wordnet
from nltk.tokenize import word_tokenize

nltk.download('opinion_lexicon')

# Load lexicon and models
positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

def find_conflicting_words(sentence):
    """
    Detect conflicting words in a sentence based on positive and negative lexicon.
    Args:
        sentence (str): Input sentence.
    Returns:
        dict: A dictionary with 'positive_words', 'negative_words', and 'is_conflict'.
    """
    words = word_tokenize(sentence.lower())
    pos_words = [word for word in words if word in positive_words]
    neg_words = [word for word in words if word in negative_words]

    return {
        "positive_words": pos_words,
        "negative_words": neg_words,
        "is_conflict": bool(pos_words and neg_words)
    }

def compute_word_sentiments(words):
    """
    Compute sentiment scores for individual words using a sentiment classification model.
    Args:
        words (list): List of words in the sentence.
    Returns:
        list of tensors: Sentiment vectors (Positive, Neutral, Negative) for each word.
    """
    sentiment_vectors = []
    for word in words:
        inputs = tokenizer(word, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        sentiment_vectors.append(torch.softmax(outputs.logits, dim=1).squeeze())
    return sentiment_vectors

def find_best_split(words, sentiment_vectors):
    """
    Find the split point in the sentence where the conflict between two parts is maximized.
    Args:
        words (list): List of words in the sentence.
        sentiment_vectors (list of tensors): Sentiment vectors for each word.
    Returns:
        tuple: Index of the split point.
    """
    max_conflict = -float("inf")
    best_split = 0

    # Iterate over possible split points
    for i in range(1, len(words)):
        left_score = sum(sentiment_vectors[:i])
        right_score = sum(sentiment_vectors[i:])
        conflict = torch.norm(left_score - right_score, p=2)
        if conflict > max_conflict:
            max_conflict = conflict
            best_split = i

    return best_split

def detect_implicit_negation(context):
    """
    Detect if the context implies negation through specific patterns or verbs.
    Args:
        context (str): Sentence or phrase containing the target word.
    Returns:
        bool: True if implicit negation is detected, False otherwise.
    """
    negation_verbs = ["pretend", "act as if", "seem", "appear to"]  # Add more implicit negation verbs
    for verb in negation_verbs:
        if verb in context.lower():
            return True
    return False

def disambiguate_word(word, context):
    """
    Disambiguate the meaning of a word in the given context.
    Args:
        word (str): Target word to disambiguate.
        context (str): Context in which the word appears.
    Returns:
        str: Word with its disambiguated meaning (or the original word if no definition is found).
    """
    # Check for implicit negation in the context
    is_negated = detect_implicit_negation(context)

    # Use BERT model for word sense disambiguation
    inputs = tokenizer(f"{word} in context: {context}", return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
    predicted_sense = torch.argmax(outputs.logits).item()

    # Retrieve sense definition from WordNet
    synsets = wordnet.synsets(word)
    if synsets and predicted_sense < len(synsets):
        sense_definition = synsets[predicted_sense].definition()

        # If implicit negation is detected, adjust the meaning
        if is_negated:
            sense_definition = f"not {sense_definition}" if not sense_definition.startswith("not") else sense_definition

        return f"{word} ({sense_definition})"

    # If no definition is found, return the original word
    return word

def split_text_based_on_conflict(input_ids, tokenizer):
    """
    Split text into Part 1 and Part 2 based on the maximum conflict point.
    Args:
        input_ids (Union[Tensor, str]): Input IDs from the tokenizer or decoded text.
        tokenizer: The tokenizer to decode input IDs back to text.
    Returns:
        tuple: (Part 1 text, Part 2 text).
    """
    # If input is a tensor, decode it to text
    if isinstance(input_ids, torch.Tensor):
        sentence = tokenizer.decode(input_ids.squeeze(), skip_special_tokens=True)
    elif isinstance(input_ids, str):
        sentence = input_ids  # Already decoded text
    else:
        raise ValueError("input_ids must be a Tensor or a string")

    words = sentence.split()

    # Step 1: Detect conflicting words
    conflicting_words = find_conflicting_words(sentence)

    # Step 2: If conflict exists, split the sentence
    if conflicting_words["is_conflict"]:
        sentiment_vectors = compute_word_sentiments(words)  # Compute word-level sentiment vectors
        split_index = find_best_split(words, sentiment_vectors)  # Find the best split point
        part1_text = " ".join(words[:split_index])
        part2_text = " ".join(words[split_index:])
    else:
        # No conflict: return the full sentence as Part 1 and an empty Part 2
        part1_text = sentence
        part2_text = ""

    return part1_text, part2_text


def process_sentences(target_data):
    """
    Process sentences by detecting conflicts, splitting into parts, and disambiguating target words.
    Args:
        target_data (list): List of dictionaries with sentence text and identified target words.
    Returns:
        list: List of processed sentences with target words disambiguated and parts split.
    """
    processed_sentences = []

    for item in target_data:
        sentence = item["text"]
        target_words = item["entities"]

        # Step 1: Detect conflicting words
        print("Step 1: Detect conflicting words")
        conflicting_words = find_conflicting_words(sentence)

        # Step 2: Split sentence if there is a conflict
        print("Step 2: Split sentence if there is a conflict")
        words = sentence.split()
        if conflicting_words["is_conflict"]:
            # Compute sentiment vectors and find best split
            sentiment_vectors = compute_word_sentiments(words)
            split_index = find_best_split(words, sentiment_vectors)
            part1, part2 = " ".join(words[:split_index]), " ".join(words[split_index:])
        else:
            # No conflict, keep the sentence as Part 1
            print("Conflicting words exist.")
            part1, part2 = sentence, ""
        print(f"Part 1: {part1}")
        print(f"Part 2: {part2}")

        # Step 3: Disambiguate target words in each part
        print("Disambiguate target words in each part")
        for word in target_words:
            if word in part1:
                part1 = part1.replace(word, f"{word} ({disambiguate_word(word, part1)})")
            if word in part2:
                part2 = part2.replace(word, f"{word} ({disambiguate_word(word, part2)})")

        # Combine parts into the final processed sentence
        processed_sentences.append(f"{part1} [SEP] {part2}" if part2 else part1)

    return processed_sentences
