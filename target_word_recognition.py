import nltk
from nltk import pos_tag, word_tokenize
nltk.download('averaged_perceptron_tagger_eng')

def perform_ner(sentences):
    """
    Identify ambiguous target words in each sentence.
    Args:
        sentences (list): List of sentences to process.
    Returns:
        list: List of dictionaries containing 'text' and 'entities' (target words).
    """
    target_data = []
    for sentence in sentences:
        tokens = word_tokenize(sentence)
        tagged = pos_tag(tokens)
        target_words = [word for word, tag in tagged if tag.startswith('NN') or tag.startswith('VB')]
        target_data.append({"text": sentence, "entities": target_words})
    return target_data
