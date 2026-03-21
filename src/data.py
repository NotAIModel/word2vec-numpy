import numpy as np

def get_training_pairs(indices, window_size=2):
    pairs = []
    for i, center in enumerate(indices):
        left = max(0, i - window_size)
        right = min(len(indices), i + window_size + 1)
        for j in range(left, right):
            if i != j:
                pairs.append((center, indices[j]))
    return np.asarray(pairs, dtype=np.int64)

def get_noise_distribution(word_counts, word2idx):
    vocab_size = len(word2idx)
    freqs = np.zeros(vocab_size)
    for word, idx in word2idx.items():
        freqs[idx] = word_counts[word]
    freqs = freqs ** 0.75
    freqs = freqs / freqs.sum()
    return freqs

def sample_negatives_batch(noise_dist, batch_pairs, k, oversample_factor=3):
    vocab_size = len(noise_dist)
    batch_pairs = np.asarray(batch_pairs, dtype=np.int64)
    batch_size = len(batch_pairs)

    forbidden = batch_pairs
    num_candidates = k * oversample_factor

    candidates = np.random.choice(
        vocab_size,
        size=(batch_size, num_candidates),
        p=noise_dist,
    )

    negatives = np.empty((batch_size, k), dtype=np.int64)

    for i in range(batch_size):
        row = candidates[i]
        center, context = forbidden[i]

        valid = row[(row != center) & (row != context)]

        if len(valid) < k:
            extra = []
            while len(valid) + len(extra) < k:
                sample = np.random.choice(vocab_size, p=noise_dist)
                if sample != center and sample != context:
                    extra.append(sample)
            valid = np.concatenate([valid, np.asarray(extra, dtype=row.dtype)])

        negatives[i] = valid[:k]

    return negatives