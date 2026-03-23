import os
from config import ARTIFACTS_DIR

from src.evaluate import load_embeddings, most_similar, cosine_similarity

emb_path = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
vocab_path = os.path.join(ARTIFACTS_DIR, "vocab.json")

emb, w2i, i2w = load_embeddings(emb_path, vocab_path)

print("most_similar('france'):", most_similar("france", emb, w2i, i2w))
print("most_similar('king'):", most_similar("king", emb, w2i, i2w))
print("most_similar('war'):", most_similar("war", emb, w2i, i2w))
print("cosine_similarity('king', 'queen'):", cosine_similarity("king", "queen", emb, w2i))