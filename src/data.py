import numpy as np

def get_training_pairs(indices, window_size=2):
    pairs = []
    for i, center in enumerate(indices):
        left = max(0, i - window_size)
        right = min(len(indices), i + window_size + 1)
        for j in range (left, right):
            if i != j:
                pairs.append((center, indices[j]))
    return pairs

def get_noise_distribution(word_counts, word2idx):
    vocab_size = len(word2idx)
    freqs = np.zeros(vocab_size)
    for word, idx in word2idx.items():
        freqs[idx] = word_counts[word]
    freqs = freqs ** 0.75
    freqs = freqs / freqs.sum()
    return freqs 

def sample_negatives(noise_dist, k, exclude):
    negatives = []
    while len(negatives) < k:
        sample = np.random.choice(len(noise_dist), p=noise_dist)
        if sample not in exclude:
            negatives.append(sample)
    return negatives     