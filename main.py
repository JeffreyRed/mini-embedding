"""
main.py — End-to-end pipeline for mini-embedding.

Usage:
    python main.py

Pipeline:
    1. Show every (target, context) training pair so you can see what the
       model actually learns from.
    2. Train the Skip-gram Word2Vec model with live epoch progress.
    3. Interactive query loop — type any word to find its nearest neighbours.
    4. Save learned embeddings to disk.
    5. Plot the 2-D embedding space and the loss curve.
"""

import torch
from pathlib import Path

from src.dataset import TextDataset
from src.model import Word2Vec
from src.train import train
from src.utils import most_similar
from src.visualize import plot_embeddings, plot_loss, animate_training, plot_matrix_updates

# ── Config ────────────────────────────────────────────────────────────────────
CORPUS_PATH   = "data/corpus.txt"
EMBEDDING_DIM = 2        # 2 for animated 2-D visualisation; try 8 for richer embeddings
WINDOW_SIZE   = 2
EPOCHS        = 150
LEARNING_RATE = 0.01
OUTPUTS_DIR   = Path("outputs")
# ──────────────────────────────────────────────────────────────────────────────


def show_training_pairs(dataset: TextDataset) -> None:
    """Prints every (target, context) pair so you can see exactly what
    the model learns from — no magic, just word co-occurrence."""
    print("── Training pairs (what the model sees) ────────────")
    print(f"  Window size : {dataset.window_size}  →  look {dataset.window_size} word(s) left AND right")
    print()

    # Group pairs back by sentence for readability
    sentences = dataset.sentences
    pair_idx = 0
    all_pairs = dataset.data

    for sentence in sentences:
        sentence_str = " ".join(sentence)
        print(f"  Sentence: \"{sentence_str}\"")

        n = len(sentence)
        for i, word in enumerate(sentence):
            for offset in range(-dataset.window_size, dataset.window_size + 1):
                if offset == 0:
                    continue
                j = i + offset
                if 0 <= j < n:
                    context_word = sentence[j]
                    direction = "→" if offset > 0 else "←"
                    print(f"    target={word:<10}  context={context_word:<10}  "
                          f"(offset {offset:+d})  {direction}")
        print()

    print(f"  Total pairs : {len(all_pairs)}")
    print("────────────────────────────────────────────────────\n")


def _resolve_word(raw: str, lower_to_vocab: dict) -> str:
    """Maps user input to the correct vocab casing, e.g. 'i' → 'I'."""
    return lower_to_vocab.get(raw.lower(), raw)


def interactive_query(model: Word2Vec, dataset: TextDataset) -> None:
    """
    Interactive nearest-neighbour lookup.
    Type a word → get its top-3 most similar words + similarity scores.
    Type 'quit' or press Ctrl-C to exit.
    """
    vocab          = list(dataset.word2idx.keys())
    lower_to_vocab = {w.lower(): w for w in vocab}   # "i" → "I", "cats" → "cats"

    print("── Interactive word search ──────────────────────────")
    print(f"  Vocabulary: {vocab}")
    print("  Type a word to find its nearest neighbours.")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            raw   = input("  Query word: ").strip()
            query = _resolve_word(raw, lower_to_vocab)
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting query mode.")
            break

        if raw.lower() in ("quit", "exit", "q"):
            break

        if query not in dataset.word2idx:
            print(f"  ✗ '{raw}' not in vocabulary. Try one of: {vocab}\n")
            continue

        results = most_similar(
            model, query, dataset.word2idx, dataset.idx2word, top_k=3
        )
        print(f"\n  '{query}'  →  nearest neighbours:")
        for rank, (word, score) in enumerate(results, 1):
            bar = "█" * int(score * 20) if score > 0 else ""
            print(f"    {rank}. {word:<12} {score:+.4f}  {bar}")
        print()

    print("────────────────────────────────────────────────────\n")


def generate_sentence(model: Word2Vec, dataset: TextDataset, length: int = 4) -> None:
    """
    Generates a sentence by chaining nearest-neighbour predictions.

    Starting from a seed word typed by the user, the model picks the
    highest-similarity neighbour at each step to form the next word.
    This is NOT how real LLMs generate text (they use probability sampling),
    but it demonstrates that the embedding space has learned directional
    structure: each word points toward its most likely companions.

    Args:
        model (Word2Vec): Trained model.
        dataset (TextDataset): Used for vocab lookup.
        length (int): Total words in the generated sentence (seed included).
    """
    vocab          = list(dataset.word2idx.keys())
    lower_to_vocab = {w.lower(): w for w in vocab}

    print("── Sentence generator ───────────────────────────────")
    print(f"  Vocabulary: {vocab}")
    print(f"  Type a seed word → the model chains {length} words using")
    print(f"  the highest-similarity neighbour at each step.")
    print("  Type 'quit' to exit.\n")

    while True:
        try:
            raw  = input("  Seed word: ").strip()
            seed = _resolve_word(raw, lower_to_vocab)
        except (KeyboardInterrupt, EOFError):
            print("\n  Exiting sentence generator.")
            break

        if raw.lower() in ("quit", "exit", "q"):
            break

        if seed not in dataset.word2idx:
            print(f"  ✗ '{raw}' not in vocabulary. Try one of: {vocab}\n")
            continue

        sentence = [seed]
        current  = seed
        used     = {seed}           # avoid immediate repetition

        for step in range(length - 1):
            neighbours = most_similar(
                model, current, dataset.word2idx, dataset.idx2word, top_k=len(vocab) - 1
            )
            # Pick the highest-scoring word not already in the sentence
            next_word = None
            for word, score in neighbours:
                if word not in used:
                    next_word = word
                    next_score = score
                    break

            if next_word is None:
                break   # vocabulary exhausted (only happens on tiny corpora)

            sentence.append(next_word)
            used.add(next_word)
            current = next_word

        print(f"\n  Generated: {' '.join(sentence)}")
        print(f"  (each arrow = highest-similarity neighbour)\n")

        # Show the chain with scores
        words = sentence
        for i in range(len(words) - 1):
            neighbours = most_similar(
                model, words[i], dataset.word2idx, dataset.idx2word, top_k=len(vocab) - 1
            )
            score = next((s for w, s in neighbours if w == words[i + 1]), 0.0)
            print(f"    {words[i]:<12} →  {words[i+1]:<12} (similarity {score:.3f})")
        print()

    print("────────────────────────────────────────────────────\n")


def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # ── 1. Load corpus ────────────────────────────────────────────────────────
    print("\n── Loading corpus ──────────────────────────────────")
    dataset = TextDataset(CORPUS_PATH, window_size=WINDOW_SIZE)
    print(dataset)
    print(f"  Vocabulary ({dataset.vocab_size} words): {list(dataset.word2idx.keys())}\n")

    # ── 2. Show training pairs ────────────────────────────────────────────────
    show_training_pairs(dataset)

    # ── 3. Build model ────────────────────────────────────────────────────────
    model = Word2Vec(vocab_size=dataset.vocab_size, embedding_dim=EMBEDDING_DIM)
    print(model, "\n")

    # ── 3b. Show embedding matrix updates (educational step) ────────────────
    print("── Embedding matrix update visualisation ───────────")
    print("  Showing how 6 gradient steps change the matrix in real time.")
    print("  Uses a SEPARATE copy of the model so training is unaffected.\n")
    import copy
    model_demo = copy.deepcopy(model)
    plot_matrix_updates(
        model_demo, dataset, dataset.idx2word,
        n_steps=6,
        save_path=str(OUTPUTS_DIR / "matrix_updates.png"),
    )

    # ── 4. Train ──────────────────────────────────────────────────────────────
    print("── Training ────────────────────────────────────────")

    # Capture embedding snapshots every N epochs for the animation
    snapshot_every = max(1, EPOCHS // 30)   # ~30 frames total
    snapshots: list = []

    loss_history = train(
        model, dataset,
        epochs=EPOCHS,
        lr=LEARNING_RATE,
        snapshots=snapshots,
        snapshot_every=snapshot_every,
    )

    # ── 5. Nearest neighbours (fixed set) ────────────────────────────────────
    fixed_queries = ["cats", "dogs", "rain", "like"]
    print("\n── Nearest neighbours (post-training) ──────────────")
    for word in fixed_queries:
        if word not in dataset.word2idx:
            continue
        results = most_similar(model, word, dataset.word2idx, dataset.idx2word, top_k=3)
        formatted = ", ".join(f"{w} ({s:.3f})" for w, s in results)
        print(f"  {word:<10}  →  {formatted}")
    print()

    # ── 6. Interactive query ──────────────────────────────────────────────────
    interactive_query(model, dataset)

    # ── 6b. Sentence generator ───────────────────────────────────────────────
    generate_sentence(model, dataset, length=4)

    # ── 7. Save embeddings ────────────────────────────────────────────────────
    save_path = OUTPUTS_DIR / "embeddings.pt"
    torch.save(model.embedding.weight.detach(), save_path)
    print(f"Embeddings saved → {save_path}\n")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    plot_embeddings(
        model, dataset.idx2word,
        save_path=str(OUTPUTS_DIR / "embeddings.png"),
    )
    plot_loss(
        loss_history,
        save_path=str(OUTPUTS_DIR / "loss_curve.png"),
    )
    if EMBEDDING_DIM == 2 and snapshots:
        animate_training(
            snapshots, dataset.idx2word,
            save_path=str(OUTPUTS_DIR / "training_animation.gif"),
        )


if __name__ == "__main__":
    main()
