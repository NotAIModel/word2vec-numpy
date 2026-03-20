import os
import json 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.vocab import tokenize, build_vocab
from src.data import get_training_pairs, get_noise_distribution, sample_negatives_batch
from src.model import Word2Vec

DATA_PATH   = "data/train.txt"
EMBED_DIM   = 100
WINDOW_SIZE = 2
NEG_SAMPLES = 5
MIN_COUNT   = 5
EPOCHS      = 2
LR          = 0.025
BATCH_SIZE  = 512
SEED = 42

def main():
    with open (DATA_PATH, "r", encoding="utf-8") as f:
        raw_text = f.read()

    np.random.seed(SEED)    
    print("tokenizing...")
    tokens = tokenize(raw_text) # ["jet", "intern", "success"]
    print(f"toatl tokens: {len(tokens):,}") # 3

    print("building vocab...")
    word2idx, idx2word, word_counts = build_vocab(tokens, min_count=MIN_COUNT) #word2idx = {"jet": 0, "intern": 1, "success": 2}
    vocab_size = len(word2idx) # 3
    print(f"vocab size: {vocab_size:,}")

    indices = [word2idx[t] for t in tokens if t in word2idx] # indices = [0, 1, 2]  
    
    print("intializing model...")
    model = Word2Vec(vocab_size=vocab_size, embed_dim=EMBED_DIM, lr=LR)
    noise_dist = get_noise_distribution(word_counts, word2idx) # dist * 0.75
    pairs = get_training_pairs(indices, window_size=WINDOW_SIZE) # (jet, intern), (jet success), (intern, jet), 
                                                                 # (intern, success), (success, jet), (success, intern)
                                                                 # but actually in numbers like
                                                                 # (0, 1), (0, 2), (1, 0)
                                                                 # (1, 2), (2, 0), (2, 1)
    loss_history = []
    for epoch in range (EPOCHS):
        total_loss = 0.0
        n_batches = 0
        np.random.shuffle(pairs)

        for i in tqdm(range(0, len(pairs), BATCH_SIZE), desc=f"epoch {epoch+1}"):
            batch = pairs[i:i+BATCH_SIZE] # batch_size = 2 |=> i=0 |=> batch =  (0, 1), (0, 2), (1, 0)

            idx_cntr = np.array([p[0] for p in batch])# 1, 2, 0
            idx_ctx  = np.array([p[1] for p in batch])# 0, 0, 1           
            idx_ctx_neg = sample_negatives_batch(noise_dist, batch, k=NEG_SAMPLES) # some numbers

            loss = model.train_step(idx_cntr, idx_ctx, idx_ctx_neg) 
            total_loss += loss
            n_batches += 1

            if n_batches == 1 or (n_batches % 1000 == 0 and n_batches > 0):
                tqdm.write(f"  batch {n_batches} | avg loss: {total_loss/n_batches:.4f}")
                loss_history.append(round(total_loss/n_batches, 4))

        print(f"epoch {epoch+1} done | avg loss: {total_loss/n_batches:.4f}")
    
    os.makedirs("artifacts", exist_ok=True)
    np.save("artifacts/embeddings.npy", model.weights_cntr)
    with open("artifacts/vocab.json", "w") as f:
        json.dump(word2idx, f)
    print("saved embeddings to artifacts/")


    with open("artifacts/loss_history.json", "w") as f:
        json.dump(loss_history, f)

    plt.figure()
    plt.plot(range(len(loss_history)), loss_history)
    plt.xlabel("batch (×1000)")
    plt.ylabel("avg loss")
    plt.title("training loss")
    plt.savefig("artifacts/loss_curve.png")
    plt.close()
    print("saved loss curve to artifacts/")

if __name__ == "__main__":
    main()