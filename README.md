# COMP550_WSD_for_Sentiment-Analysis

## Manuscript: https://www.overleaf.com/4567438473tfmtdwsrwkpq#9e551b

- **Requirement**: 
  - The length of your report must be between 4.5 to 5 pages of content. This includes text, figures, appendices, and equations. 
  - References, however, do not count towards the length limit.

- **Manuscript Reference Template**: https://arxiv.org/pdf/1804.09301

## Responsibility:
- Dalian: Recognizing words to be disambiguated.
  - Which word tends to be ambiguous? How do you handle it? Is NER word really ambiguous, please pay attention to the methods used.

  report:
  - 3-5 relevant papers, with a focus on how our work differs from theirs

- Fengfei: Disambiguate words, including implicit negation, symbol effects.
  - Motivation: Sentiment analysis fails on input with implicit negation, how to handle it? Does exclamation mark help disambiguation? How to handle self-conflicting/noisy inputs?

  report:
  - What conclusions can be drawn from your experiments?
  - initial hypothesis verified?

- Mandy: Sentiment Analysis
  - Fine-tuned BERT-based model
  - Prepare domain-specific datasets (e.g., product reviews, customer feedback).
  Fine-tune BERT on these datasets to adapt to the specific nuances of the domain.
  - Few-Shot Learning for Low-Resource Domains:
  Experiment with few-shot learning models (e.g., GPT-3, T5, or GPT-4).
  Provide a small number of labeled examples for niche domains or rare entities. (eg. shopping reviews vs medical reviews)
  - Cross-Domain Transfer Learning:
  Train the model on a large dataset from one domain (e.g., social media posts).
  Transfer learned patterns and apply the model to another domain (e.g., news articles).
  Validate performance on cross-domain datasets to measure adaptability.

  report:
  - Abstract
  - Introduction
  - results of experiment: Measure accuracy and F1 scores for sentiment predictions
  - Compare results across baseline models (BERT without fine-tune)
  - Limitations
  - Work extension
