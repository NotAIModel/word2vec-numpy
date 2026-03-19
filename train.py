import os
import json 
import numpy as np
from tqdm import tqdm

from src.vocab import tokenize, build_vocab
from src.data import get_training_pairs, get_noise_distribution, sample_negatives
from src.model import Word2Vec

DATA_PATH   = "data/train.txt"
EMBED_DIM   = 100
WINDOW_SIZE = 2
NEG_SAMPLES = 5
MIN_COUNT   = 5
EPOCHS      = 3
LR          = 0.025
BATCH_SIZE = 512  

def main():
    with open (DATA_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()
    
    print("tokenizing...")
    tokens = tokenize(raw_text)
    print(f"toatl tokens: {len(tokens):,}")

    print("building vocab...")
    word2idx, idx2word, word_counts = build_vocab(tokens, min_count=MIN_COUNT)
    vocab_size = len(word2idx)
    print(f"vocab size: {vocab_size:,}")

    indices = [word2idx[t] for t in tokens if t in word2idx]
    
    print("intializing model...")
    model = Word2Vec(vocab_size=vocab_size, embed_dim=EMBED_DIM, lr=LR)
    noise_dist = get_noise_distribution(word_counts, word2idx)
    pairs = get_training_pairs(indices, window_size=WINDOW_SIZE)

    for epoch in range (EPOCHS):
        total_loss = 0.0
        n_batches = 0

        for i in tqdm(range(0, len(pairs), BATCH_SIZE), desc=f"epoch {epoch+1}"):
            batch = pairs[i:i+BATCH_SIZE]
            
            idx_cntr    = []
            idx_ctx     = []
            idx_ctx_neg = []

            idx_cntr = [p[0] for p in batch]
            idx_ctx  = [p[1] for p in batch]

            # sample all negatives at once — B*k*3 samples in one numpy call
            candidates = np.random.choice(len(noise_dist), 
                                        size=len(batch) * NEG_SAMPLES * 3,
                                        p=noise_dist)
            idx_ctx_neg = candidates.reshape(-1, NEG_SAMPLES * 3)[:, :NEG_SAMPLES].tolist()

            loss = model.train_step(idx_cntr, idx_ctx, idx_ctx_neg)
            total_loss += loss
            n_batches += 1

            if n_batches % 100 == 0 and n_batches > 0:
                tqdm.write(f"  batch {n_batches} | avg loss: {total_loss/n_batches:.4f}")

        print(f"epoch {epoch+1} done | avg loss: {total_loss/n_batches:.4f}")
    
    os.makedirs("artifacts", exist_ok=True)
    np.save("artifacts/embeddings.npy", model.weights_cntr)
    with open("artifacts/vocab.json", "w") as f:
        json.dump(word2idx, f)
    print("saved embeddings to artifacts/")

if __name__ == "__main__":
    main()