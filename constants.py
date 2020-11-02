# File: constants
# ---------------
# Houses the constants for the experiments. Could
# be better to use config yaml files for this since
# we'll have multiple methods.

# MAML constants

# Optimization arguments
NUM_TRAIN_ITERATIONS = 15000
NUM_INNER_UPDATES = 1
OUTER_LR = 0.003
INNER_LR = 0.003

# Model architecture arguments
EMBED_DIM = 128
HIDDEN_DIM = 128

# Data loading arguments
NUM_SUPPORT = 7
NUM_QUERY = 7
META_BATCH_SIZE = 16
NUM_WORKERS = 4
CONTEXT_LEN = 12
TEST_PREFIX_LEN = 12

# Splits
TRAIN_SPLIT = ["Vocal", "Folk", "Pop_Rock", "International", "Electronic", "New Age"]
VAL_SPLIT = ["RnB", "Blues", "Latin"]
TEST_SPLIT = ["Country", "Reggae", "Jazz"]

# Miscellaneous evaluation and checkpointing arguments
MODEL_TYPES = ["SimpleLSTM"]
EVALUATE_EVERY = 100
REPORT_TRAIN_EVERY = 50
SAVE_CHECKPOINT_EVERY = 1000