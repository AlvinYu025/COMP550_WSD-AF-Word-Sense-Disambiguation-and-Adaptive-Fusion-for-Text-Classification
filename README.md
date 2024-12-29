# COMP550 WSD-AF for Sentiment Analysis

Welcome! This repository contains the official code for the final project of COMP550: Natural Language Processing at McGill University. Below, you'll find an overview of our project.

---

## Motivation

![Motivation](src/motivation.pdf)

In everyday life, we encounter many noisy sentences with hidden semantics that can be challenging to understand, even for highly educated individuals. These noisy inputs may come from young children, people with speech disabilities, or individuals with autism. Accurately extracting sentiments from these inputs is crucial for better understanding and addressing the needs of these groups, potentially contributing to solutions for various social issues.

Our project addresses the limitations of text classification models in adversarial and noisy contexts. Specifically:

- **Implicit Negation**: Sentences like "I pretend to be happy" contain subtle negations that current deep neural network (DNN) models struggle to classify correctly. BERT-based models, even after fine-tuning, often fail to capture the nuanced text distribution of such inputs.

- **Self-Conflicting Semantics**: Inputs with conflicting semantics (e.g., "I love the service, but the food was terrible.") confuse classifiers, leading to degraded performance. These scenarios highlight the need for improved methods to identify true sentiments amidst noise.

---

## Method: WSD-AF

![WSD-AF Framework](src/framework.pdf)

To address these challenges, we propose **WSD-AF**, a method combining adaptive fusion and word sense disambiguation:

1. **Adaptive Fusion (AF)**: A weight generation network (Fusion Model) adaptively assigns weights to:
   - **Global Semantic Vector**: Captures the overall sentiment of the input.
   - **Context Semantic Vector**: Focuses on sentences with core components.
   This fusion enhances classification performance in adversarial contexts by balancing these two semantic vectors.

2. **Word Sense Disambiguation (WSD)**: By integrating WSD, we provide richer context, enabling the model to reduce noise and better recognize explicit semantics in noisy sentences.

---

## Running the Code

The project includes four evaluation modes: **Baseline**, **WSD**, **AF**, and **WSD-AF**. Switch between modes freely to explore their effects.

### Steps to Execute

1. **Load and Preprocess Sentiment Model Dataset**
    - We use two datasets for our experiments:
        - **Sentiment Analysis Dataset (SAD)**
        - **Hospital Review Dataset (HRD)**
    - Run the following command:
    ```bash
    python SAD_main.py
    ```
    or
    ```bash
    python HRD_main.py
    ```

2. **Fine-Tune the BERT-Based Classifier**
    - We fine-tune **BERT-base-uncased** and **RoBERTa-base** models.
    - Modify the model name in the corresponding Python file if needed.

3. **Load and Process Fusion Dataset**
    - The Fusion dataset (MC dataset) consists of:
        - 150 training samples
        - 60 test samples
    - These samples were generated using ChatGPT and manually crafted.

4. **Combine Sentiment and Fusion Datasets for Evaluation**
    - To ensure fair comparisons, we augment the sentiment datasets with the MC dataset, simulating noisy inputs.

5. **Train Alpha Calculator (for "AF" or "WSD-AF" Modes)**
    - Uncomment the relevant code in the script if you wish to evaluate using "AF" or "WSD-AF" modes.

6. **Final Evaluation**
    - Evaluate the methods (Baseline, WSD, AF, WSD-AF) based on metrics like accuracy, recall, precision, and F1-score.

---

## Contributions

### Mandy Huang
- Co-authored the abstract and introduction.
- Designed the experimental setup, including models, datasets, metrics, and fine-tuning parameters.
- Implemented the model fine-tuning code and handled dataset cleaning/loading.
- Presented experimental results, including model performance and limitations.

### Fengfei Yu
- Led the project, including manuscript and code development.
- Co-authored the abstract, motivation, and WSD-AF framework.
- Designed and implemented the fusion model, disambiguation-related code, and dependency parsing.
- Proposed techniques for handling noisy/conflicting inputs and created all visual figures.

### Danlin Luo
- Authored the related work section and contributed to the introduction.
- Focused on training and fine-tuning models on the HRD dataset.
- Implemented target word recognition code and managed paper formatting.

---

We hope this README provides clarity on the project and helps you navigate the codebase. Happy exploring!
