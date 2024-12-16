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
  - Abstract (Fengfei)
  - Introduction
  - Related work (short)
  - Motivation (Fengfei)
    - Motivation 1: Current Model cannot identify complex syntax, such as implicit negation: "I pretend to be happy." -> This is due to lack of disambiguation. -> Thus we add disambiguation to improve the Sentiment Analysis/Review (classification) performance.
    - Motivation 2: Current Model cannot handle noisy sentences with self-conflicting meanings in one sentence: "I love this poor, stubborn, and ugly man." -> We propose a divide-and-conquer method with dynamic weighting to solve this problem. (alpha calculator)
  - Experiment Setup
    - What model used.
      - BERT + FT BERT + RoBERTa + FT RoBERTa (FT: Fine-tuned)
    - What dataset used.
      - Sentiment Analysis Dataset + Tweet + Product Review + Hospital Review (?)
    - Evaluation metrics used. (ACC + F1)
    - Parameter setting (fine-tune epoches/learning rate).
  - Experiment Results
    - Refer to Picture.
  - Discussion
    - Compare results across baseline models (w/ or w/o fine-tune)
    - Analyze the effect of WSD
    - Analyze the effect of DaC (Divide-and-Conquer)
    - Analyze the effect of WSDaC (WSD + DaC)
  - Limitations
    - Advanced splitting method can be explored. (Ours is based on sentiment word only.)
    - Dataset limitation: We haven't test long sentences with alternating sentiments, sometimes indicated by time. The noisy dataset is limited for your testing.
    - Extra cost of training alpha calculator.
  - Work extension
