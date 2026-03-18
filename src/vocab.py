import re 
from collections import Counter

def tokenize(text):
    text = text.lower()
    tokens = re.findall(r'\b[a-z]+\b', text)
    return tokens

def build_vocab(tokens, min_count=5):
    counts = Counter(tokens)
    counts = {w: c for w, c in counts.items() if c >= min_count}

    word2idx = {w: i for i, w in enumerate(counts.keys())}
    idx2word = {i: w for w, i in word2idx.items()}
    return word2idx, idx2word, counts