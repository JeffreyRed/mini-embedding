"""mini-embedding — source package."""
from src.dataset import TextDataset
from src.model import Word2Vec
from src.train import train
from src.utils import most_similar, print_similar
from src.visualize import plot_embeddings, plot_loss

__all__ = [
    "TextDataset",
    "Word2Vec",
    "train",
    "most_similar",
    "print_similar",
    "plot_embeddings",
    "plot_loss",
]
