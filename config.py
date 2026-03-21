import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(ROOT_DIR, "data", "train.txt")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

EMBED_DIM = 100
WINDOW_SIZE = 2
NEG_SAMPLES = 5
MIN_COUNT = 5
EPOCHS = 2

# backward() uses summed gradients, so changing BATCH_SIZE changes the effective step size.
# If you change BATCH_SIZE substantially, retune LR as well.
LR = 0.025
BATCH_SIZE = 512

SEED = 42
