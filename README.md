# word2vec-numpy
Word2Vec skip-gram with negative sampling, implemented from scratch in pure NumPy. 
Built as a JetBrains internship application task.

## What's implemented

- Tokenization and vocabulary building with min-frequency filtering
- Skip-gram pair generation with sliding window
- Negative sampling with unigram distribution smoothed by frequency^0.75
- Batched forward pass with manual gradient derivation
- SGD updates via `np.add.at` for correct repeated-index accumulation
- Gradient correctness verified with finite-difference tests
- Cosine similarity and most-similar word evaluation

## Results

Trained on WikiText-2 (~1.6M tokens, vocab ~19k words), 2 epochs. Loss mostly plateaued after epoch 2, additional epochs showed diminishing returns without learning rate decay.

![training loss](artifacts/loss_curve.png)

Some nearest neighbors after training:
```
most_similar("france")              → italy (0.884), germany (0.864), russia (0.834)
most_similar("king")                → henry (0.796), vi (0.793), queen (0.783)
most_similar("war")                 → civil (0.768), outbreak (0.756), conflict (0.710)
cosine_similarity("king", "queen")  → 0.783
```


## Usage
```bash
pip install -r requirements.txt
python train.py
```

Pre-trained embeddings are included in `artifacts/` (I've uploaded them as they weigh tiny) 

You can skip training and evaluate directly:

```bash

python -c "
from src.evaluate import load_embeddings, most_similar, cosine_similarity

emb, w2i, i2w = load_embeddings('artifacts/embeddings.npy', 'artifacts/vocab.json')
print(most_similar('france', emb, w2i, i2w))
print(cosine_similarity('king', 'queen', emb, w2i))
"
```
Or just modify `tests/demo.py` with your examples and run it.

```bash
python tests/demo.py
```

## Design notes

- **Gradient scaling**: loss is reported as batch mean for readability, but 
  `backward()` computes summed gradients (no `/B` division). This is a deliberate 
  choice. Dividing gradients by B would require scaling LR up by B to compensate:

  | reduction | formula | equivalent LR |
  |-----------|---------|---------------|
  | sum (ours) | `step = lr * B * mean_grad` | `lr = 0.025` |
  | mean | `step = lr * mean_grad` | `lr = 0.025 * B = 12.8` |

  Both are mathematically equivalent for constant-LR SGD. We prefer summed 
  gradients + `lr=0.025` since 0.025 is the value from the original paper and 
  avoids an unintuitive large LR in the config. Note: this coupling means 
  changing `BATCH_SIZE` requires adjusting `LR` proportionally.

- **Two embedding matrices**: `weights_cntr` (center) and `weights_ctx` (context),
  both shape `(V, D)`. Final embeddings use `weights_cntr` — standard practice 
  since it's updated more directly by the training signal.
- **Embedding dim = 100**: original paper used 300 on a much larger corpus.
  100 is sufficient for WikiText-2.
- **Batch size = 512**: single numpy call per batch for negative sampling,
  vectorized matmul for scores. ~120 it/s on CPU.
- **Tokenizer**: lowercase letters only (`[a-z]+`). Simple and sufficient for 
  WikiText-2, though it drops punctuation and numbers.

## Possible improvements

- Learning rate decay over epochs
- Frequent-word subsampling (the original paper's trick for better rare-word embeddings)
- Dynamic window size — sample window uniformly from [1, max_window]
- Evaluate on word analogy benchmarks (Google analogy dataset)
- Stream training pairs instead of materializing all 6M into memory