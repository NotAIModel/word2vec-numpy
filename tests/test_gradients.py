import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.model import Word2Vec


def compute_loss(model, idx_cntr, idx_ctx, idx_ctx_neg):
    loss, *_ = model.forward(idx_cntr, idx_ctx, idx_ctx_neg)
    return loss


def numerical_gradient(model, idx_cntr, idx_ctx, idx_ctx_neg,
                       matrix_name, row, col, eps=1e-5):
    """Centered finite-difference gradient of the mean loss w.r.t. one parameter."""
    matrix = getattr(model, matrix_name)
    orig = matrix[row, col]

    matrix[row, col] = orig + eps
    loss_plus = compute_loss(model, idx_cntr, idx_ctx, idx_ctx_neg)

    matrix[row, col] = orig - eps
    loss_minus = compute_loss(model, idx_cntr, idx_ctx, idx_ctx_neg)

    matrix[row, col] = orig
    return (loss_plus - loss_minus) / (2 * eps)


def make_model(vocab_size=6, embed_dim=3, lr=0.01):
    """Create a model with nonzero weights_ctx so all gradients are nonzero."""
    model = Word2Vec(vocab_size=vocab_size, embed_dim=embed_dim, lr=lr)
    model.weights_ctx = np.random.uniform(-0.1, 0.1, (vocab_size, embed_dim))
    return model


def test_train_step_updates():
    """
    Verify that train_step() produces weight changes matching numerical gradients.

    backward() uses sum-reduced gradients (no 1/B division), so the expected
    weight change is:  Δw = -lr * B * dL_mean/dw
    """
    np.random.seed(42)

    vocab_size = 6
    embed_dim = 3
    lr = 0.01

    # duplicate center index 1 to exercise np.add.at accumulation
    idx_cntr = [1, 2, 1]
    idx_ctx = [3, 4, 0]
    idx_ctx_neg = [[0, 2], [0, 1], [2, 5]]
    B = len(idx_cntr)

    params_to_test = [
        ("weights_cntr", 1, 0),   # center word appearing twice — tests accumulation
        ("weights_cntr", 2, 1),   # center word appearing once
        ("weights_ctx",  3, 0),   # positive-only context word
        ("weights_ctx",  0, 2),   # appears as both positive ctx and negative ctx
        ("weights_ctx",  5, 1),   # negative-only context word
    ]

    for matrix_name, row, col in params_to_test:
        np.random.seed(42)
        model = make_model(vocab_size, embed_dim, lr)

        num_grad = numerical_gradient(
            model, idx_cntr, idx_ctx, idx_ctx_neg, matrix_name, row, col
        )

        w_before = getattr(model, matrix_name)[row, col]
        model.train_step(idx_cntr, idx_ctx, idx_ctx_neg)
        w_after = getattr(model, matrix_name)[row, col]

        actual_delta = w_after - w_before
        expected_delta = -lr * B * num_grad

        assert np.allclose(actual_delta, expected_delta, atol=1e-7), (
            f"{matrix_name}[{row},{col}]: "
            f"actual Δw={actual_delta:.10f}, expected Δw={expected_delta:.10f}"
        )


def test_untouched_weights_unchanged():
    """Weights not referenced by any index must not change."""
    np.random.seed(42)
    model = make_model()

    idx_cntr = [0]
    idx_ctx = [1]
    idx_ctx_neg = [[2]]

    cntr_before = model.weights_cntr.copy()
    ctx_before = model.weights_ctx.copy()

    model.train_step(idx_cntr, idx_ctx, idx_ctx_neg)

    # rows 3, 4, 5 are not referenced anywhere
    for row in [3, 4, 5]:
        assert np.array_equal(model.weights_cntr[row], cntr_before[row]), (
            f"weights_cntr[{row}] changed but was not in any index"
        )
        assert np.array_equal(model.weights_ctx[row], ctx_before[row]), (
            f"weights_ctx[{row}] changed but was not in any index"
        )


if __name__ == "__main__":
    test_train_step_updates()
    test_untouched_weights_unchanged()
    print("All gradient tests passed.")
