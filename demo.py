import os
import numpy as np
import matplotlib.pyplot as plt

from config import ARTIFACTS_DIR
from src.evaluate import load_embeddings, most_similar, cosine_similarity


def pca_2d(vectors):
    vectors = vectors - vectors.mean(axis=0)
    cov = np.cov(vectors.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    idx = np.argsort(eigenvalues)[::-1][:2]
    components = eigenvectors[:, idx]
    return vectors @ components


def main():
    emb_path   = os.path.join(ARTIFACTS_DIR, "embeddings.npy")
    vocab_path = os.path.join(ARTIFACTS_DIR, "vocab.json")
    emb, w2i, i2w = load_embeddings(emb_path, vocab_path)

    # --- text output ---
    queries = ['france', 'king', 'war']
    results = {}
    for q in queries:
        results[q] = most_similar(q, emb, w2i, i2w)
        print(f"most_similar('{q}'): {results[q]}")
    print(f"cosine_similarity('king', 'queen'): {cosine_similarity('king', 'queen', emb, w2i)}")

    # --- visualization ---
    # build groups from query results
    colors = ['red', 'blue', 'green']
    groups = {}
    for q, color in zip(queries, colors):
        group_words = [q] + [w for w, _ in results[q]]
        groups[q] = (color, [w for w in group_words if w in w2i])

    all_words = list({w for _, (_, words) in groups.items() for w in words})
    vectors = np.array([emb[w2i[w]] for w in all_words])
    coords = pca_2d(vectors)
    word_to_coord = {w: coords[i] for i, w in enumerate(all_words)}

    fig, ax = plt.subplots(figsize=(11, 8))

    for group, (color, words) in groups.items():
        for w in words:
            if w not in word_to_coord:
                continue
            x, y = word_to_coord[w]
            ax.scatter(x, y, color=color, s=60, zorder=3, label=group)
            ax.annotate(w, (x, y), textcoords="offset points",
                        xytext=(5, 5), fontsize=9)
        # arrow from query word to each of its neighbors
        if group in word_to_coord:
            for w, _ in results[group]:
                if w in word_to_coord:
                    x1, y1 = word_to_coord[group]
                    x2, y2 = word_to_coord[w]
                    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                                arrowprops=dict(arrowstyle="->", color=color,
                                                lw=0.8, alpha=0.5))

    handles, labels = ax.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax.legend(seen.values(), seen.keys(), fontsize=9)

    ax.set_title("PCA projection — query words and their nearest neighbors", fontsize=11)
    ax.axhline(0, color='gray', lw=0.3)
    ax.axvline(0, color='gray', lw=0.3)
    plt.tight_layout()

    out_path = os.path.join(ARTIFACTS_DIR, "embedding_viz.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nsaved visualization to {out_path}")


if __name__ == "__main__":
    main()