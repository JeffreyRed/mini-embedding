# Theory & Code Walkthrough — mini-embedding

> Step 1 of the mini-LLM series: learning what embeddings are, why they work, and exactly what every line of code does.

---

## Table of Contents

1. [The problem embeddings solve](#1-the-problem-embeddings-solve)
2. [What an embedding actually is](#2-what-an-embedding-actually-is)
3. [The Skip-gram objective](#3-the-skip-gram-objective)
4. [How meaning emerges from training](#4-how-meaning-emerges-from-training)
5. [Measuring similarity — cosine distance](#5-measuring-similarity--cosine-distance)
6. [The embedding IS the neural network — clearing up the confusion](#6-the-embedding-is-the-neural-network--clearing-up-the-confusion)
7. [Code walkthrough](#7-code-walkthrough)
   - [dataset.py](#datasetpy)
   - [model.py](#modelpy)
   - [train.py](#trainpy)
   - [utils.py](#utilspy)
   - [visualize.py](#visualizepy)
   - [main.py](#mainpy)
8. [The full data flow](#8-the-full-data-flow)
9. [Connection to LLMs](#9-connection-to-llms)

---

## 1. The problem embeddings solve

Computers cannot work directly with words. Every word must be turned into a number — or better, a list of numbers — before any math can happen.

The naïve approach is to assign each word an integer ID:

```
"cat"  →  1
"dog"  →  2
"rain" →  3
```

This has a fatal problem: the numbers carry **no information about meaning**. The model sees that `cat = 1` and `dog = 2`, which implies they are just as different as `cat = 1` and `rain = 3`. There is no sense in which `cat` and `dog` are closer to each other than to `rain`.

A slightly better approach is **one-hot encoding**: give each word a vector of zeros with a single `1` at its index position.

```
Vocabulary: ["cat", "dog", "rain"]   (size = 3)

"cat"  →  [1, 0, 0]
"dog"  →  [0, 1, 0]
"rain" →  [0, 0, 1]
```

Still broken. Every pair of words is equidistant — the dot product between any two one-hot vectors is always zero. And in a real vocabulary of 50,000 words, each vector would have 50,000 dimensions, almost all of them zero. Computationally wasteful, semantically empty.

**Embeddings fix both problems at once.**

---

## 2. What an embedding actually is

An embedding is a **trainable matrix** `E` of shape `(vocab_size × embedding_dim)`:

```
         dim_0   dim_1   dim_2  ...  dim_d
"cat"  [ 0.12,  -0.70,   0.33, ...,  0.91 ]
"dog"  [ 0.14,  -0.65,   0.31, ...,  0.88 ]
"rain" [-0.60,   0.10,  -0.42, ...,  0.05 ]
  ⋮
```

Each row is the **embedding vector** for one word — a dense, low-dimensional representation that the model learns during training.

The lookup operation is trivial: given word index `i`, return row `i` of `E`. In PyTorch this is exactly what `nn.Embedding` does — it is nothing more than a differentiable lookup table.

```python
E = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d)
vector = E(torch.tensor([2]))   # returns row 2 of the matrix
```

The matrix starts with **random values**. The interesting question is: how does training turn random noise into meaningful vectors?

---

## 3. The Skip-gram objective

The intuition comes from linguistics:

> *"You shall know a word by the company it keeps."*  — J.R. Firth, 1957

Words that appear near each other in text tend to be related in meaning. The **Skip-gram** model exploits this directly: given a target word, predict which words are likely to appear nearby.

**Step 1 — Build a sliding window over the corpus.**

Take the sentence `"I like cats"` with window size `2`:

```
Target: "like"   →   Context: ["I", "cats"]
Target: "I"      →   Context: ["like"]
Target: "cats"   →   Context: ["like", "I"]
```

Every `(target, context)` pair becomes one training example.

**Step 2 — Frame it as a classification problem.**

For each target word, the model must predict the correct context word out of the entire vocabulary. This is a multi-class classification over `vocab_size` classes.

```
Input:  index of "like"   →   Model   →   probability distribution over all words
                                           P("I")    = 0.31  ✓ high (correct context)
                                           P("cats") = 0.28  ✓ high (correct context)
                                           P("rain") = 0.02  ✗ low  (never co-occurs)
```

**Step 3 — Loss: cross-entropy.**

Cross-entropy penalises the model when the predicted probability for the correct context word is low. After each training step, the embedding vectors for co-occurring words are nudged toward each other; vectors for non-co-occurring words drift apart.

```
Loss = -log( P(correct context word) )
```

Low loss → model assigns high probability to true context words → embeddings of related words are close together.

---

## 4. How meaning emerges from training

Consider the sentences in `data/corpus.txt`:

```
I like cats
I like dogs
dogs and cats are animals
I hate rain
```

After training:

- `"cats"` and `"dogs"` both appear after `"like"` and before `"animals"` — their embeddings converge.
- `"like"` and `"love"` appear with similar neighbours — their embeddings converge.
- `"hate"` and `"rain"` co-occur together but never with `"like"` — they cluster separately.

No human ever labeled `cats` and `dogs` as similar. The model discovered this purely from co-occurrence statistics. This is the core mechanism behind every word vector model, from Word2Vec to the token embeddings inside GPT-4.

---

## 5. Measuring similarity — cosine distance

After training, we compare word vectors using **cosine similarity**:

```
         A · B
sim =  ─────────
        |A| |B|
```

- Result of `1.0` → vectors point in exactly the same direction → maximum similarity
- Result of `0.0` → vectors are orthogonal → unrelated
- Result of `-1.0` → vectors point in opposite directions → antonyms

Cosine similarity is preferred over Euclidean distance because it measures **direction**, not magnitude. Two words can have vectors of different lengths but still be semantically equivalent if they point the same way.

```python
# Example from utils.py
cosine_similarity(embed("cats"), embed("dogs"))  →  ~0.97
cosine_similarity(embed("cats"), embed("rain"))  →  ~0.10
```

---

## 6. The embedding IS the neural network — clearing up the confusion

A common source of confusion: it sounds like the embedding is a *separate process* that happens before the neural network. It is not. The embedding matrix **is** the first layer of the neural network — initialized with random values, updated by backpropagation, just like any other weight matrix.

The full picture of what you have after training:

```
WHAT YOU HAVE AFTER TRAINING
─────────────────────────────────────────────────────────────────

  Input:  word index  (e.g. 4 = "cats")
              │
              ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Embedding Matrix  E  (vocab_size × emb_dim)            │ ← WEIGHTS
  │                                    updated by backprop  │
  │  "I"       →  [ 0.12,  0.34]  ← row 0                  │
  │  "and"     →  [-0.21,  0.11]  ← row 1                  │
  │  "cats"    →  [ 0.91,  0.80]  ← row 4  ← used          │
  │  "dogs"    →  [ 0.88,  0.76]  ← row 5                  │
  │  "rain"    →  [-0.60, -0.42]  ← row 8                  │
  │  ...                                                    │
  └─────────────────────────────────────────────────────────┘
              │  returns row 4 = [0.91, 0.80]
              ▼
  ┌─────────────────────────────────────────────────────────┐
  │  Linear Layer  W  (emb_dim × vocab_size)                │ ← also WEIGHTS
  └─────────────────────────────────────────────────────────┘
              │  returns [0.1, 0.3, 1.8, 1.7, -0.9, ...]
              │          (one score per vocabulary word)
              ▼
  ┌─────────────────────────────────────────────────────────┐
  │  CrossEntropyLoss                                       │
  │  = softmax → probabilities → -log(correct one)         │
  └─────────────────────────────────────────────────────────┘
              │
              ▼  scalar loss  (e.g. 2.34)
              │
         backprop + Adam update both E and W
```

### What each piece is doing

**Embedding Matrix E** — this is not a lookup table that was built separately. It is a `(vocab_size × embedding_dim)` weight matrix, the same kind of object as any `nn.Linear` layer, except that instead of a matrix multiplication it does a row-index lookup. That lookup is equivalent to multiplying a one-hot vector by `E`, which is just selecting one row. PyTorch makes this efficient with `nn.Embedding`.

**Linear Layer W** — projects the embedding vector back up to `vocab_size` dimensions, producing one raw score (logit) per word. This is what makes the task a classification problem: which of the `vocab_size` words is the correct context?

**CrossEntropyLoss** — applies softmax to the logits (turning them into probabilities that sum to 1), then computes `-log(P(correct context word))`. If the model assigns high probability to the correct context word, the loss is low. If it guesses wrong, the loss is high.

**Backpropagation** — PyTorch automatically differentiates through the whole chain and computes `∂loss / ∂E` and `∂loss / ∂W`. The gradient for `E` only has non-zero values in the row corresponding to the target word — every other row gets a zero gradient for that step.

**Adam optimizer** — applies the gradients with an adaptive learning rate:

```
E[target_row]  ←  E[target_row]  -  lr × gradient
```

A concrete example for one step with `("like" → "cats")`:

```
Before:  E["like"] = [-0.12,  0.44]
Gradient computed by backprop:  [-0.03, +0.07]

After:   E["like"] = [-0.12 - 0.01×(-0.03),   0.44 - 0.01×(+0.07)]
                   = [-0.1197,  0.4393]
```

Only row `"like"` moves. Row `"cats"` gets its own update when it appears as a target in a different pair. Over thousands of steps, every row drifts to a position where it can predict its typical neighbours — and that drift is what we call "learning meaning".

### What the output actually is

A point of confusion: the output of the model during training is **one probability distribution** over the vocabulary — not a matrix of similar words. The similarity search (`most_similar()`) is a completely separate step that happens *after* training, by directly comparing rows of `E` using cosine similarity. The linear layer `W` is discarded entirely for inference.

| Phase | What's used | Output |
|---|---|---|
| Training | E + W + Loss | Scalar loss → gradients → weight updates |
| Inference | E only | Cosine similarity between rows → nearest neighbours |

### Summary: answering the exact question

| Question | Answer |
|---|---|
| Does the embedding "become" a neural network? | It always was one — `nn.Embedding` is a weight matrix from line 1 |
| We init with random values, then train? | Yes — random init, then backprop adjusts every value in E |
| The matrix is the first layer's weights? | Exactly correct |
| The output is a matrix of closest vectors? | No — during training the output is a probability distribution. Closest-word search is a post-training query on rows of E directly |

---

## 7. Code walkthrough

### `dataset.py`

**Responsibility:** read the corpus, build a vocabulary, generate all `(target, context)` training pairs.

```python
class TextDataset:
    def __init__(self, path: str, window_size: int = 2) -> None:
```

The constructor takes the path to `corpus.txt` and a window size (how many words left/right of the target count as context).

---

```python
    def _build_vocab(self) -> None:
        words = [w for sentence in self.sentences for w in sentence]
        vocab = sorted(set(words))
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
```

Flattens all sentences into a single word list, deduplicates with `set()`, sorts alphabetically for reproducibility, then builds two dictionaries:

- `word2idx` — `"cats" → 4`
- `idx2word` — `4 → "cats"`

---

```python
    def _build_training_pairs(self) -> List[Tuple[int, int]]:
        pairs = []
        for sentence in self.sentences:
            for i, word in enumerate(sentence):
                target_idx = self.word2idx[word]
                for offset in range(-self.window_size, self.window_size + 1):
                    if offset == 0:
                        continue
                    j = i + offset
                    if 0 <= j < n:
                        context_idx = self.word2idx[sentence[j]]
                        pairs.append((target_idx, context_idx))
```

For every word in every sentence, the inner loop slides a window of `[-window_size, +window_size]` around it, skipping offset `0` (the word itself) and skipping out-of-bounds positions. Each valid neighbour produces one `(target_idx, context_idx)` tuple.

For `"I like cats"` with `window_size=2` this generates:

```
(I, like), (like, I), (like, cats), (cats, like), (cats, I)
```

---

```python
    def get_batches(self):
        for target, context in self.data:
            yield torch.tensor([target]), torch.tensor([context])
```

A generator that converts each integer pair into a PyTorch tensor on the fly. Using `yield` means the entire dataset never needs to be in memory as tensors at once.

---

### `model.py`

**Responsibility:** define the neural network architecture.

```python
class Word2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.linear    = nn.Linear(embedding_dim, vocab_size)
        nn.init.kaiming_uniform_(self.linear.weight)
```

Two layers:

| Layer | Shape | Role |
|---|---|---|
| `nn.Embedding` | `vocab_size × embedding_dim` | The lookup table we are training |
| `nn.Linear` | `embedding_dim × vocab_size` | Projects the vector back to a score over all words |

`kaiming_uniform_` initialises the linear weights with values scaled to the layer size, which helps gradients flow stably at the start of training.

---

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(x)   # (batch, embedding_dim)
        logits   = self.linear(embedded) # (batch, vocab_size)
        return logits
```

The forward pass is just two operations:

1. **Embedding lookup** — given the word index, retrieve its row from the matrix `E`. Shape goes from `(1,)` to `(1, embedding_dim)`.
2. **Linear projection** — multiply by a weight matrix to get one score per vocabulary word. Shape goes from `(1, embedding_dim)` to `(1, vocab_size)`.

These raw scores (logits) are passed directly to the loss function, which applies softmax internally.

---

```python
    def get_embedding(self, word_idx: int) -> torch.Tensor:
        with torch.no_grad():
            return self.embedding.weight[word_idx]
```

A convenience method used after training to retrieve a word vector without computing gradients. `torch.no_grad()` is important here — we are doing inference, not training, so there is no need to build a computational graph.

---

### `train.py`

**Responsibility:** run the training loop.

```python
def train(model, dataset, epochs=100, lr=0.01, verbose=True) -> list:
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
```

- **CrossEntropyLoss** — combines `softmax` + `negative log-likelihood` in one numerically stable operation. It compares the model's logits against the true context word index.
- **Adam** — an adaptive gradient descent optimizer. It tracks a running average of past gradients to adjust the learning rate per-parameter, which makes it much faster to converge than plain SGD on this kind of task.

---

```python
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for target, context in dataset.get_batches():
            optimizer.zero_grad()          # clear gradients from last step
            logits = model(target)         # forward pass
            loss   = loss_fn(logits, context)  # compute loss
            loss.backward()                # backpropagate gradients
            optimizer.step()               # update all weights
            total_loss += loss.item()
```

This is the standard PyTorch training loop. The five steps — `zero_grad → forward → loss → backward → step` — repeat for every training pair in every epoch.

`loss.backward()` is where backpropagation runs. PyTorch automatically computes `∂loss/∂W` for every parameter `W` in the model — including every value in the embedding matrix. `optimizer.step()` then applies those gradients to nudge each parameter slightly in the direction that reduces loss.

---

### `utils.py`

**Responsibility:** cosine similarity and nearest-neighbor search.

```python
def cosine_similarity(vec1, vec2) -> float:
    return F.cosine_similarity(vec1.unsqueeze(0), vec2.unsqueeze(0)).item()
```

`unsqueeze(0)` adds a batch dimension because `F.cosine_similarity` expects 2-D inputs. `.item()` converts the result from a 1-element tensor to a plain Python float.

---

```python
def most_similar(model, word, word2idx, idx2word, top_k=3):
    target_vec = model.get_embedding(word2idx[word])
    similarities = []
    for idx in range(len(word2idx)):
        if idx == target_idx:
            continue
        sim = cosine_similarity(target_vec, model.get_embedding(idx))
        similarities.append((idx2word[idx], sim))
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:top_k]
```

Brute-force nearest-neighbor search: compare the query vector against every other word in the vocabulary, sort by similarity, return the top `k`. For a 16-word vocabulary this is trivial; for 50,000 words you would use an approximate search library (FAISS, Annoy). The logic is identical.

---

### `visualize.py`

**Responsibility:** plot the embedding space and the training loss curve.

```python
def _get_2d_embeddings(model) -> np.ndarray:
    weights = model.embedding.weight.detach().numpy()
    if weights.shape[1] == 2:
        return weights
    # PCA to 2-D
    weights_centered = weights - weights.mean(axis=0)
    _, _, Vt = np.linalg.svd(weights_centered, full_matrices=False)
    return weights_centered @ Vt[:2].T
```

If `embedding_dim=2` (as in a demo), no reduction is needed. Otherwise, **PCA (Principal Component Analysis)** is applied: the Singular Value Decomposition (SVD) finds the two directions of maximum variance in the high-dimensional space and projects all vectors onto them. This preserves the large-scale geometric structure — clusters that are far apart in 50-D will still be far apart in 2-D.

`detach()` is required before calling `.numpy()` because PyTorch tensors that are attached to the computational graph cannot be converted directly.

---

### `main.py`

**Responsibility:** tie everything together in one runnable script.

```python
# Config block at the top — all hyperparameters in one place
EMBEDDING_DIM  = 8
WINDOW_SIZE    = 2
EPOCHS         = 150
LEARNING_RATE  = 0.01
```

Keeping all configuration at the top of the entry point is a clean practice: you never need to dig into source files to tune a parameter.

The execution flow is strictly linear:

```
Load corpus → Build vocab → Init model → Train → Evaluate → Save → Plot
```

Each step calls exactly one function from `src/`, making `main.py` a readable summary of the whole pipeline.

---

## 8. The full data flow

Tracing a single training example end-to-end:

```
corpus.txt
  "I like cats"
        │
        ▼  dataset.py
  Vocabulary: {I:0, and:1, animals:2, are:3, cats:4, ...}
  Pair generated: ("like" → idx 9,  "cats" → idx 4)
        │
        ▼  model.py  forward()
  embedding(tensor([9]))
     → row 9 of E  →  [-0.12, 0.44, 0.71, ..., 0.03]   shape: (1, 8)
  linear([-0.12, 0.44, ...])
     → [0.1, -0.3, 0.8, 0.5, 1.2, ...]                  shape: (1, 16)
        │
        ▼  train.py
  CrossEntropyLoss( logits, target=4 )
     → softmax → -log(P(cats)) → scalar loss
        │
        ▼  loss.backward()
  Gradient flows back through linear layer
  into embedding matrix row 9 ("like")
  and, through the loss signal, nudges row 4 ("cats")
  to become more predictable from row 9
        │
        ▼  optimizer.step()
  Row 9 ("like") and Row 4 ("cats") updated
  They become slightly more similar
```

Repeat 180 pairs × 150 epochs = 27,000 updates. By the end, the geometry of the 16×8 matrix encodes the semantic structure of the corpus.

---

## 9. Connection to LLMs

This project implements the exact same mechanism used in the input layer of GPT, BERT, and every other transformer-based language model. The differences are scale and context:

| Property | mini-embedding | GPT-3 |
|---|---|---|
| Vocabulary size | ~16 | 50,257 |
| Embedding dimension | 8 | 12,288 |
| Training pairs | ~200 | ~300 billion tokens |
| Objective | Skip-gram | Next-token prediction |
| What follows the embedding | Linear layer | 96 transformer blocks |

The embedding matrix in GPT is trained jointly with the attention layers, not separately as here. But the object is identical: a `(vocab_size × embedding_dim)` matrix, updated by gradient descent, that maps discrete tokens into a continuous geometric space where proximity encodes meaning.

Understanding mini-embedding means you already understand the first layer of every LLM that exists.

---

*Next: `mini-attention` — scaled dot-product self-attention from scratch, taking these embedding vectors as input.*