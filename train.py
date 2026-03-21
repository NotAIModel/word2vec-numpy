import os
import json

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from config import (
    ARTIFACTS_DIR, BATCH_SIZE, DATA_PATH, EMBED_DIM,
    EPOCHS, LR, MIN_COUNT, NEG_SAMPLES, SEED, WINDOW_SIZE,
)
from src.data import get_noise_distribution, get_training_pairs, sample_negatives_batch
from src.model import Word2Vec
from src.vocab import build_vocab, tokenize

def train(model, pairs, noise_dist, epochs, batch_size, neg_samples):
    loss_history = []

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        np.random.shuffle(pairs)

        for i in tqdm(range(0, len(pairs), batch_size), desc=f"epoch {epoch + 1}"):
            batch = pairs[i:i + batch_size]

            idx_cntr = batch[:, 0]
            idx_ctx = batch[:, 1]
            idx_ctx_neg = sample_negatives_batch(noise_dist, batch, k=neg_samples)

            loss = model.train_step(idx_cntr, idx_ctx, idx_ctx_neg)
            total_loss += loss
            n_batches += 1

            if n_batches == 1 or n_batches % 1000 == 0:
                avg_loss = total_loss / n_batches
                tqdm.write(f"  batch {n_batches} | avg loss: {avg_loss:.4f}")
                loss_history.append(round(avg_loss, 4))

        print(f"epoch {epoch + 1} done | avg loss: {total_loss / n_batches:.4f}")

    return loss_history


def save_artifacts(model, word2idx, loss_history, artifacts_dir):
    os.makedirs(artifacts_dir, exist_ok=True)

    np.save(os.path.join(artifacts_dir, "embeddings.npy"), model.weights_cntr)
    
    with open(os.path.join(artifacts_dir, "vocab.json"), "w") as f:
        json.dump(word2idx, f)
    
    with open(os.path.join(artifacts_dir, "loss_history.json"), "w") as f:
        json.dump(loss_history, f)

    plt.figure()
    plt.plot(range(len(loss_history)), loss_history)
    plt.xlabel("batch (×1000)")
    plt.ylabel("avg loss")
    plt.title("training loss")
    plt.savefig(os.path.join(artifacts_dir, "loss_curve.png"))
    plt.close()

    print(f"saved artifacts to {artifacts_dir}/")

def main():
    np.random.seed(SEED)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    print("tokenizing...")
    tokens = tokenize(raw_text)
    print(f"total tokens: {len(tokens):,}")

    print("building vocab...")
    word2idx, idx2word, word_counts = build_vocab(tokens, min_count=MIN_COUNT)
    vocab_size = len(word2idx)
    print(f"vocab size: {vocab_size:,}")

    indices = [word2idx[t] for t in tokens if t in word2idx]
    pairs = get_training_pairs(indices, window_size=WINDOW_SIZE)

    print("initializing model...")
    model = Word2Vec(vocab_size=vocab_size, embed_dim=EMBED_DIM, lr=LR)
    noise_dist = get_noise_distribution(word_counts, word2idx)

    loss_history = train(model, pairs, noise_dist, EPOCHS, BATCH_SIZE, NEG_SAMPLES)
    save_artifacts(model, word2idx, loss_history, ARTIFACTS_DIR)

if __name__ == "__main__":
    main()