from nltk.corpus import wordnet

# Perform word sense disambiguation on target words and modify sentences
def disambiguate(texts, target_data):
    modified_sentences = []

    for item in target_data:
        original_text = item["text"]
        target_words = item["entities"]

        # Disambiguate each target word using WordNet
        for word in target_words:
            # Find all synsets for the word
            synsets = wordnet.synsets(word)
            if synsets:
                # Select the first synset as an example (you can refine this logic)
                best_synset = synsets[0]
                sense_description = best_synset.definition()

                # Modify the original text by appending sense description to the target word
                original_text = original_text.replace(word, f"{word} ({sense_description})")

        modified_sentences.append(original_text)

    return modified_sentences
