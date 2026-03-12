# RNN Sentiment Analysis on IMDB

A PyTorch-based repository for sentiment classification on the IMDB movie review dataset using recurrent neural models.

## Overview

This project explores sequence models for binary sentiment classification (`positive` vs `negative`) on IMDB reviews:

- A baseline **vanilla RNN** workflow in [src/IMDB.ipynb](src/IMDB.ipynb) via [`TextRNN`](src/IMDB.ipynb)
- A stronger **bidirectional multi-layer LSTM** workflow in [src/LSTM.ipynb](src/LSTM.ipynb) via [`SentimentClassifier`](src/LSTM.ipynb)
- Optional pretrained **GloVe embeddings** loaded with `torchtext` in [src/LSTM.ipynb](src/LSTM.ipynb)
- Early stopping + checkpointing to `best_parameters.pth`

## Repository Structure

- [README.md](README.md): project documentation
- [LICENSE](LICENSE): MIT license
- [src/IMDB.ipynb](src/IMDB.ipynb): vanilla RNN sentiment model notebook
- [src/LSTM.ipynb](src/LSTM.ipynb): BiLSTM + GloVe sentiment model notebook
- [src/.gitignore](src/.gitignore): ignore rules

## Models

### 1) Vanilla RNN (IMDB.ipynb)
Defined by [`TextRNN`](src/IMDB.ipynb):
- Embedding layer
- `nn.RNN` encoder
- Final linear classifier
- Trained with `BCEWithLogitsLoss`

### 2) Bidirectional LSTM (LSTM.ipynb)
Defined by [`SentimentClassifier`](src/LSTM.ipynb):
- Embedding initialized from GloVe (200d)
- 2-layer bidirectional `nn.LSTM`
- Dropout + linear head
- Trained with AdamW + cosine annealing
- Early stopping on validation loss
- Best checkpoint persisted as `best_parameters.pth`

## Data Pipeline

Both notebooks use:
- `datasets.load_dataset("imdb")`
- basic English tokenization via `torchtext`
- custom collate function with dynamic padding:
  - [`collate_batch`](src/IMDB.ipynb) in RNN notebook
  - [`collate`](src/LSTM.ipynb) in LSTM notebook
- max sequence truncation to 256 tokens during batching

## Environment Setup

Use Python 3.10+ (3.11 used in notebook metadata).

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install datasets torchtext tqdm jupyter
```

> Note: CUDA is optional. The notebooks already fall back to CPU.

## How to Run

### Option A: Vanilla RNN experiment
1. Open [src/IMDB.ipynb](src/IMDB.ipynb)
2. Run cells top to bottom:
   - data loading
   - vocab creation
   - model definition
   - training loop
   - inference via [`predict_statement`](src/IMDB.ipynb)

### Option B: LSTM + GloVe experiment
1. Open [src/LSTM.ipynb](src/LSTM.ipynb)
2. Run cells top to bottom:
   - data loading and vocab
   - GloVe initialization
   - model setup
   - train/validation loops via [`train_step`](src/LSTM.ipynb) and [`val_step`](src/LSTM.ipynb)
   - checkpoint load and inference via [`predict_sentiment`](src/LSTM.ipynb)
   - test accuracy/confusion matrix

## Reported Notebook Outcomes

From current notebook outputs in [src/LSTM.ipynb](src/LSTM.ipynb):
- Early stopping around epoch 9
- Test accuracy printed around **86.568%**

From [src/IMDB.ipynb](src/IMDB.ipynb):
- Training loss decreases steadily over epochs
- Inference examples run directly after training

## Notes and Caveats

- `torchtext` currently emits deprecation warnings in notebook output.
- AMP (`autocast`, `GradScaler`) is enabled in code but auto-disables on CPU.
- Thresholding should be applied on sigmoid probabilities:
  - use `torch.sigmoid(logits) >= 0.5` for class prediction
- Notebooks are research-style workflows; packaging into `.py` modules is a possible next step., and `numpy`

## Reproducibility Tips

- Set random seeds in `torch`, `random`
- Save vocabulary mapping alongside model weights
- Keep train/validation split deterministic (`generator` in `random_split`)
- Version-pin dependencies in a requirements file

## Future Improvements

- Move code from notebooks into reusable modules under `src/`
- Add metrics: precision, recall, F1, ROC-AUC
- Add experiment tracking (e.g., TensorBoard/W&B)
- Add unit tests for tokenization and collate functions

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).