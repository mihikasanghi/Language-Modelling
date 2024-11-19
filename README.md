# ANLP Assignment 1 - Language Models Implementation

## Overview
This repository contains implementations of three different language models:
1. Neural Network Language Model (NNLM) - 5-gram based
2. LSTM-based Language Model
3. Transformer Decoder-based Language Model

## Dataset
- Corpus: Auguste_Maquet
- Data split:
  - Training: 70%
  - Validation: 10%
  - Test: 20%

## Requirements
- Python 3.x
- PyTorch (primary Deep Learning framework)
- Additional libraries:
  - Scikit-learn (for handling pre-trained embeddings)
  - Gensim (for word embeddings)

## Model Architectures

### 1. Neural Network Language Model (NNLM)
- Input: Pre-trained embeddings of previous 5 words (5-gram context)
- Architecture:
  - Input Layer: Concatenated 5-gram embeddings
  - Hidden Layer 1: 300-dimensional output
  - Hidden Layer 2: Vocabulary-size output
  - Softmax Layer: Probability distribution over vocabulary

### 2. LSTM Language Model
- Architecture:
  - LSTM layers with hidden states and cell states
  - Input: Word embeddings (300-dimensional)
  - Linear layer: Transforms to vocabulary size
  - Softmax layer: Probability distribution
- Maximum sequence length: 20 (due to compute limitations)

### 3. Transformer Decoder
- Based on the architecture from "Attention Is All You Need"
- Maximum sequence length: 20 (due to compute limitations)

## Instructions to Run the Code

```bash
# Run Neural Network Language Model
python SourceCode/NNLM.py

# Run LSTM Language Model
python SourceCode/LSTM.py

# Run Transformer Language Model
python SourceCode/TransformerLM.py
```

## Model Performance

### NNLM
Average Perplexity:
- Training: 518.7
- Validation: 667.8
- Testing: 651.3

### LSTM
Average Perplexity:
- Training: 65.1
- Validation: 352.8
- Testing: 361.0

### Transformer
Average Perplexity:
- Training: 834.6
- Validation: 861.9
- Testing: 887.1

## Analysis

### Model Performance Comparison
1. Performance Hierarchy:
   - Best: LSTM (lowest perplexity)
   - Middle: NNLM
   - Last: Transformer

2. Key Observations:
   - LSTM outperforms other models despite its simpler architecture
   - Transformer's underperformance might be due to limited dataset size
   - NNLM shows expected performance given its limited context window

### Overfitting Analysis
All models show similar training patterns:
- Training loss: Continuously decreases
- Validation loss: Initially decreases, then rises
- Solution: Model selection based on best validation performance

## Output Format
Perplexity scores are saved in text files with the following format:
- Format: `Sentence TAB perplexity-score`
- File naming: `LM<num>-train-perplexity.txt`, `LM<num>-test-perplexity.txt`, `LM<num>-val-perplexity.txt`
- Average score included at the end of each file

## References
1. [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
3. [Recurrent Neural Network Based Language Model](citation-needed)
4. [A Neural Probabilistic Language Model](citation-needed)