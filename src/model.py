import numpy as np

# NOTE: MAKES TRAINING SEVERAL TIMES FASTER, THOUGH NOT SURE IF NUMBA ACCEPTABLE FOR THIS TASK

# from numba import njit

# @njit(cache=True)
# def scatter_add(weights, indices, grads, lr):
#     for i in range(indices.shape[0]):
#         for d in range(grads.shape[1]):
#             weights[indices[i], d] -= lr * grads[i, d]  # fused, no temp array

# @njit(cache=True)
# def scatter_add_2d(weights, indices_2d, grads_3d, lr):
#     B, K = indices_2d.shape
#     D = grads_3d.shape[2]
#     for i in range(B):
#         for j in range(K):
#             for d in range(D):
#                 weights[indices_2d[i, j], d] -= lr * grads_3d[i, j, d]

def sigmoid(x):
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
        
        emb_cntr = self.weights_cntr[idx_cntr]                  # (B, D)
        emb_ctx_pos = self.weights_ctx[idx_ctx]                 # (B, D)
        emb_ctx_neg = self.weights_ctx[idx_ctx_neg]             # (B, K, D)

        # score_pos = np.sum(emb_cntr * emb_ctx_pos, axis=1)          # (B,)
        # score_neg = np.matmul(emb_ctx_neg, emb_cntr[:, :, None]).squeeze(-1)  # (B,k)
        
        score_pos = np.einsum('bd,bd->b', emb_cntr, emb_ctx_pos)
        score_neg = np.einsum('bkd,bd->bk', emb_ctx_neg, emb_cntr)
        loss = -np.mean(log_sigmoid(score_pos) + np.sum(log_sigmoid(-score_neg), axis=1))

        return loss, emb_cntr, emb_ctx_pos, emb_ctx_neg, score_pos, score_neg

    def backward(self, idx_cntr, idx_ctx, idx_ctx_neg,
                emb_cntr, emb_ctx_pos, emb_ctx_neg, score_pos, score_neg):
        d_pos = ((sigmoid(score_pos) - 1.0))[:, None]   # intentionally not dividing by B, to avoid  
        d_neg = ((sigmoid(score_neg)))[:, :, None]      # setting big LR, see README for the details

        # grad_emb_cntr = d_pos * emb_ctx_pos + np.sum(d_neg * emb_ctx_neg, axis=1)
        grad_emb_cntr = d_pos * emb_ctx_pos + np.einsum('bkd,bk->bd', emb_ctx_neg, d_neg.squeeze(-1))
        grad_emb_ctx_pos = d_pos * emb_cntr
        grad_emb_ctx_neg = d_neg * emb_cntr[:, None, :]

        np.add.at(self.weights_cntr, idx_cntr, -self.lr * grad_emb_cntr)
        np.add.at(self.weights_ctx, idx_ctx, -self.lr * grad_emb_ctx_pos)
        np.add.at(self.weights_ctx, idx_ctx_neg, -self.lr * grad_emb_ctx_neg)
    
        # scatter_add(self.weights_cntr, idx_cntr, grad_emb_cntr, self.lr)
        # scatter_add(self.weights_ctx,  idx_ctx, grad_emb_ctx_pos, self.lr)
        # scatter_add_2d(self.weights_ctx, idx_ctx_neg, grad_emb_ctx_neg, self.lr)
    
    def train_step(self, idx_cntr, idx_ctx, idx_ctx_neg):
        loss, emb_cntr, emb_ctx_pos, emb_ctx_neg, score_pos, score_neg = self.forward(
            idx_cntr, idx_ctx, idx_ctx_neg
        )
        self.backward(
            idx_cntr, idx_ctx, idx_ctx_neg,
            emb_cntr, emb_ctx_pos, emb_ctx_neg, 
            score_pos, score_neg)
        return loss