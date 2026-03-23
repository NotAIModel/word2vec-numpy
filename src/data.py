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

#NOTE: Works faster  0.2862s vs 7.5835s for this dataset on my machine
# But ignores corpus boundaries, so in the *training.py* "get_training_pairs" is selected 
def get_training_pairs_fast(indices, window_size=2):
    indices = np.asarray(indices, dtype=np.int64)
    n = len(indices)
    
    offsets = [j for j in range(-window_size, window_size + 1) if j != 0]
    
    all_centers = []
    all_contexts = []
    
    for offset in offsets:
        if offset > 0:
            centers = indices[:n - offset]
            contexts = indices[offset:]
        else:
            centers = indices[-offset:]
            contexts = indices[:n + offset]
        all_centers.append(centers)
        all_contexts.append(contexts)
    
    centers = np.concatenate(all_centers)
    contexts = np.concatenate(all_contexts)
    return np.stack([centers, contexts], axis=1)

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
    batch_size = len(batch_pairs)

    candidates = np.random.choice(
        vocab_size, size=(batch_size, k * oversample_factor), p=noise_dist
    )

    centers = batch_pairs[:, 0:1]   # (B, 1)
    contexts = batch_pairs[:, 1:2]   # (B, 1)
    mask = (candidates != centers) & (candidates != contexts)  # (B, num_candidates)

    cumsum = np.cumsum(mask, axis=1)
    selected = mask & (cumsum <= k)
    order = np.argsort(~selected, axis=1, kind='stable')
    negatives = np.take_along_axis(candidates, order, axis=1)[:, :k]

    # Fallback only for rows that didn't get k valid (rare but safe)
    valid_count = cumsum[:, -1]
    for i in np.where(valid_count < k)[0]:
        center, context = int(batch_pairs[i, 0]), int(batch_pairs[i, 1])
        have = int(valid_count[i])
        extra = []
        while have + len(extra) < k:
            s = int(np.random.choice(vocab_size, p=noise_dist))
            if s != center and s != context:
                extra.append(s)
        negatives[i, have:] = extra

    return negatives