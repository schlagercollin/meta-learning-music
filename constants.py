# File: constants
# ---------------
# Houses the constants for the experiments. Could
# be better to use config yaml files for this since
# we'll have multiple methods.

# MAML constants

# Optimization arguments
NUM_TRAIN_ITERATIONS = 15000
NUM_INNER_UPDATES = 1
OUTER_LR = 0.001
INNER_LR = 0.003

# Baseline arguments
BASELINE_NUM_EPOCHS = 10
BASELINE_LR = 0.003
BASELINE_BATCH_SIZE = 1
BASELINE_VAL_EVERY = 10000
BASELINE_REPORT_TRAIN_EVERY = 500

# Model architecture arguments
EMBED_DIM = 128
HIDDEN_DIM = 128
NUM_BLOCKS = 3
NUM_HEADS = 8
VOCAB_SIZE = 128

# Data loading arguments
NUM_SUPPORT = 5
NUM_QUERY = 5
META_BATCH_SIZE = 32
NUM_WORKERS = 4
CONTEXT_LEN = 120
TEST_PREFIX_LEN = 0

# Data sampling arguments
GENERATION_LENGTH = 120
CONDITION_LENGTH = CONTEXT_LEN
TEMPERATURE = 1.0

# Splits
TRAIN_SPLIT = ["Vocal", "Folk", "Pop_Rock", "International", "Electronic", "New Age"]
VAL_SPLIT = ["RnB", "Blues", "Latin"]
TEST_SPLIT = ["Country", "Reggae", "Jazz"]

# Miscellaneous evaluation and checkpointing arguments
MODEL_TYPES = ["SimpleLSTM", "SimpleTransformer"]
EVALUATE_EVERY = 100
REPORT_TRAIN_EVERY = 50
SAVE_CHECKPOINT_EVERY = 1000
TESTING_ITERATIONS = 1000
