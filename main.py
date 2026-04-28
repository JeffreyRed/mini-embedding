"""
main.py — End-to-end demo of mini-embedding.

Usage:
    python main.py

What it does:
    1. Loads and tokenizes the corpus
    2. Trains a Skip-gram Word2Vec model
    3. Prints nearest neighbors for selected query words
    4. Saves the learned embeddings to disk
    5. Plots the 2-D embedding space and loss curve
"""

import torch
from pathlib import Path

from src.dataset import TextDataset
from src.model import Word2Vec
from src.train import train
from src.utils import print_similar
from src.visualize import plot_embeddings, plot_loss

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_PATH    = "data/corpus.txt"
EMBEDDING_DIM  = 8       # try 2 for direct 2-D visualization
WINDOW_SIZE    = 2
EPOCHS         = 150
LEARNING_RATE  = 0.01
QUERY_WORDS    = ["cats", "dogs", "rain", "like"]
OUTPUTS_DIR    = Path("outputs")
# ──────────────────────────────────────────────────────────────────────────────


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # 1. Dataset
    print("── Loading corpus ──────────────────────────────────")
    dataset = TextDataset(CORPUS_PATH, window_size=WINDOW_SIZE)
    print(dataset)
    print(f"  Vocabulary : {list(dataset.word2idx.keys())}")
    print()

    # 2. Model
    model = Word2Vec(vocab_size=dataset.vocab_size, embedding_dim=EMBEDDING_DIM)
    print(model, "\n")

    # 3. Training
    print("── Training ────────────────────────────────────────")
    loss_history = train(model, dataset, epochs=EPOCHS, lr=LEARNING_RATE)

    # 4. Nearest neighbors
    print_similar(model, QUERY_WORDS, dataset.word2idx, dataset.idx2word)

    # 5. Save embeddings
    save_path = OUTPUTS_DIR / "embeddings.pt"
    torch.save(model.embedding.weight.detach(), save_path)
    print(f"Embeddings saved → {save_path}\n")

    # 6. Visualize
    plot_embeddings(model, dataset.idx2word, save_path=str(OUTPUTS_DIR / "embeddings.png"))
    plot_loss(loss_history, save_path=str(OUTPUTS_DIR / "loss_curve.png"))


if __name__ == "__main__":
    main()
