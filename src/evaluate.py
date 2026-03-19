import numpy as np
import json

def load_embeddings(emb_path, vocab_path):
    embeddings = np.load(emb_path)
    with open(vocab_path, "r") as f:
        word2idx = json.load(f)
    idx2word = {i: w for w, i in word2idx.items()}
    return embeddings, word2idx, idx2word

def cosine_similarity(word1, word2, embeddings, word2idx):
    v1 = embeddings[word2idx[word1]]
    v2 = embeddings[word2idx[word2]]
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    
def most_similar(word, embeddings, word2idx, idx2word, top_n=5):
    if word not in word2idx:
        print(f"'{word}' not in vocab")
        return[]
    
    vec = embeddings[word2idx[word]]
    sims = embeddings.dot(vec) / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(vec) + 1e-8)
    sims[word2idx[word]] = -1
    top_indices = np.argsort(sims)[::-1][:top_n]
    return [(idx2word[i], round(sims[i], 4)) for i in top_indices]