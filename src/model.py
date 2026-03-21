import numpy as np

def sigmoid(x):
    x = np.asarray(x)
    out = np.empty_like(x, dtype=np.float64)

    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))

    exp_x = np.exp(x[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)

    return out

def log_sigmoid(x):
    return -np.logaddexp(0.0, -x)


class Word2Vec:
    def __init__(self, vocab_size, embed_dim=100, lr=0.025):
        self.lr = lr
        limit = 0.5 / embed_dim
        self.weights_cntr = np.random.uniform(-limit, limit, (vocab_size, embed_dim))
        self.weights_ctx = np.random.uniform(-limit, limit, (vocab_size, embed_dim))

    def forward(self, idx_cntr, idx_ctx, idx_ctx_neg):
        idx_cntr = np.asarray(idx_cntr, dtype=np.int64)
        idx_ctx = np.asarray(idx_ctx, dtype=np.int64)
        idx_ctx_neg = np.asarray(idx_ctx_neg, dtype=np.int64)
        
        emb_cntr = self.weights_cntr[idx_cntr]                  # (B, D)
        emb_ctx_pos = self.weights_ctx[idx_ctx]                 # (B, D)
        emb_ctx_neg = self.weights_ctx[idx_ctx_neg]             # (B, K, D)

        score_pos = np.sum(emb_cntr * emb_ctx_pos, axis=1)          # (B,)
        score_neg = np.matmul(emb_ctx_neg, emb_cntr[:, :, None]).squeeze(-1)  # (B,k)

        loss = -np.mean(log_sigmoid(score_pos) + np.sum(log_sigmoid(-score_neg), axis=1))

        return loss, emb_cntr, emb_ctx_pos, emb_ctx_neg, score_pos, score_neg

    def backward(self, idx_cntr, idx_ctx, idx_ctx_neg,
                emb_cntr, emb_ctx_pos, emb_ctx_neg, score_pos, score_neg):
        d_pos = ((sigmoid(score_pos) - 1.0))[:, None]   # intentionally not dividing by B, to avoid  
        d_neg = ((sigmoid(score_neg)))[:, :, None]      # setting big LR, see README for the details

        grad_emb_cntr = d_pos * emb_ctx_pos + np.sum(d_neg * emb_ctx_neg, axis=1)
        grad_emb_ctx_pos = d_pos * emb_cntr
        grad_emb_ctx_neg = d_neg * emb_cntr[:, None, :]

        np.add.at(self.weights_cntr, idx_cntr, -self.lr * grad_emb_cntr)
        np.add.at(self.weights_ctx, idx_ctx, -self.lr * grad_emb_ctx_pos)
        np.add.at(self.weights_ctx, idx_ctx_neg, -self.lr * grad_emb_ctx_neg)
    
    def train_step(self, idx_cntr, idx_ctx, idx_ctx_neg):
        loss, emb_cntr, emb_ctx_pos, emb_ctx_neg, score_pos, score_neg = self.forward(
            idx_cntr, idx_ctx, idx_ctx_neg
        )
        self.backward(
            idx_cntr, idx_ctx, idx_ctx_neg,
            emb_cntr, emb_ctx_pos, emb_ctx_neg, 
            score_pos, score_neg)
        return loss