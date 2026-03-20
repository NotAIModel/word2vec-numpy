import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

class Word2Vec:
    def __init__(self, vocab_size, embed_dim=100, lr=0.025):
        self.lr = lr
        self.weights_cntr = np.random.uniform(-0.5/embed_dim, 0.5/embed_dim, (vocab_size, embed_dim))
        self.weights_ctx = np.zeros((vocab_size, embed_dim)) 
        # zero init for weights_ctx — on step 1, weights_cntr also gets zero gradient
        # (since grad_cntr depends on weights_ctx values). both matrices start learning
        # from step 2 onward once weights_ctx gets its first update.

    def forward(self, idx_cntr, idx_ctx, idx_ctx_neg):
        emb_cntr     = self.weights_cntr[idx_cntr]
        emb_ctx_pos  = self.weights_ctx[idx_ctx]
        emb_ctx_neg  = self.weights_ctx[idx_ctx_neg]

        score_pos = np.sum(emb_cntr * emb_ctx_pos, axis=1)          # (B,)
        score_neg = np.matmul(emb_ctx_neg, emb_cntr[:, :, None]).squeeze(-1)  # (B,k)

        loss = -np.mean(np.log(sigmoid(score_pos) + 1e-7)) \
               -np.mean(np.sum(np.log(sigmoid(-score_neg) + 1e-7), axis=1))

        return loss, emb_cntr, emb_ctx_pos, emb_ctx_neg, score_pos, score_neg

    def backward(self, idx_cntr, idx_ctx, idx_ctx_neg,
                emb_cntr, emb_ctx_pos, emb_ctx_neg, score_pos, score_neg):
        d_pos = ((sigmoid(score_pos) - 1))[:, None]     # intentionally not dividing by B, to avoid  
        d_neg = ((sigmoid(score_neg)))[:, :, None]      # setting big LR, see README for the details

        grad_emb_cntr    = d_pos * emb_ctx_pos + np.sum(d_neg * emb_ctx_neg, axis=1)
        grad_emb_ctx_pos = d_pos * emb_cntr
        grad_emb_ctx_neg = d_neg * emb_cntr[:, None, :]

        np.add.at(self.weights_cntr, idx_cntr,    - self.lr * grad_emb_cntr)
        np.add.at(self.weights_ctx,  idx_ctx,     - self.lr * grad_emb_ctx_pos)
        np.add.at(self.weights_ctx,  idx_ctx_neg, - self.lr * grad_emb_ctx_neg)
    
    def train_step(self, idx_cntr, idx_ctx, idx_ctx_neg):
        loss, emb_cntr, emb_ctx_pos, emb_ctx_neg, score_pos, score_neg = self.forward(
            idx_cntr, idx_ctx, idx_ctx_neg)
        self.backward(
            idx_cntr, idx_ctx, idx_ctx_neg,
            emb_cntr, emb_ctx_pos, emb_ctx_neg, 
            score_pos, score_neg)
        return loss