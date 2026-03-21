import numpy as np
import pytest
from src.model import Word2Vec

# Helpers
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


@pytest.fixture
def model():
    """Word2Vec with nonzero weights_ctx so all gradients are nonzero from step 1."""
    np.random.seed(42)
    m = Word2Vec(vocab_size=6, embed_dim=3, lr=0.01)
    m.weights_ctx = np.random.uniform(-0.1, 0.1, (6, 3))
    return m


# Tests
@pytest.mark.parametrize("matrix_name,row,col", [
    ("weights_cntr", 1, 0),  # center word appearing twice — tests np.add.at accumulation
    ("weights_cntr", 2, 1),  # center word appearing once
    ("weights_ctx",  3, 0),  # positive-only context word
    ("weights_ctx",  0, 2),  # appears as both positive and negative context
    ("weights_ctx",  5, 1),  # negative-only context word
])
def test_train_step_weight_updates(matrix_name, row, col):
    """
    Verify that train_step() produces weight changes matching numerical gradients.

    backward() uses sum-reduced gradients (no 1/B division), so the expected
    weight change is:  Δw = -lr * B * dL_mean/dw
    """
    np.random.seed(42)
    m = Word2Vec(vocab_size=6, embed_dim=3, lr=0.01)
    m.weights_ctx = np.random.uniform(-0.1, 0.1, (6, 3))

    # duplicate center index 1 to exercise np.add.at accumulation
    idx_cntr = [1, 2, 1]
    idx_ctx = [3, 4, 0]
    idx_ctx_neg = [[0, 2], [0, 1], [2, 5]]
    B = len(idx_cntr)

    num_grad = numerical_gradient(m, idx_cntr, idx_ctx, idx_ctx_neg, matrix_name, row, col)

    w_before = getattr(m, matrix_name)[row, col]
    m.train_step(idx_cntr, idx_ctx, idx_ctx_neg)
    w_after = getattr(m, matrix_name)[row, col]

    actual_delta = w_after - w_before
    expected_delta = -m.lr * B * num_grad

    assert np.allclose(actual_delta, expected_delta, atol=1e-7), (
        f"{matrix_name}[{row},{col}]: "
        f"actual Δw={actual_delta:.10f}, expected Δw={expected_delta:.10f}"
    )


def test_untouched_weights_unchanged(model):
    """Weights not referenced by any index must not change after train_step."""
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
