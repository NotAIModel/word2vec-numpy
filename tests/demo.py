import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.evaluate import load_embeddings, most_similar, cosine_similarity

emb, w2i, i2w = load_embeddings('artifacts/embeddings.npy', 'artifacts/vocab.json')

print("most_similar('france'):", most_similar('france', emb, w2i, i2w))
print("most_similar('king'):",   most_similar('king',   emb, w2i, i2w))
print("most_similar('war'):",    most_similar('war',    emb, w2i, i2w))
print("cosine_similarity('king', 'queen'):", cosine_similarity('king', 'queen', emb, w2i))