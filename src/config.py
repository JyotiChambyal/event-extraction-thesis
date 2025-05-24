# src/config.py

import os

# --- General Project Settings ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

# --- Data Paths ---
TRAIN_DATA_DIR = os.path.join(DATA_DIR, "train")
DEV_DATA_DIR = os.path.join(DATA_DIR, "dev")
TEST_DATA_DIR = os.path.join(DATA_DIR, "test")

# --- Model & Tokenizer Settings ---
BERT_MODEL_NAME = "bert-base-uncased" # Or "roberta-large", "bert-large-uncased", etc.
MAX_SEQ_LENGTH = 512 # Maximum sequence length for BERT input

# Special tokens as defined by the paper and discussion
ADDITIONAL_SPECIAL_TOKENS = ['[TYPE]', '[TRG]', '[/TRG]', '[CONTEXT]']

# --- Training Hyperparameters ---
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 2e-5 # General learning rate for BERT weights
CRF_LR_MULTIPLIER = 10 # Optional: Multiplier for CRF layer's learning rate (often higher)
ADAM_EPSILON = 1e-8
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 0 # Number of warmup steps for learning rate scheduler
GRADIENT_ACCUMULATION_STEPS = 1 # Accumulate gradients over multiple batches

# --- Loss Function Settings ---
# Set to True to enable class weighting for imbalance
CLASS_WEIGHTS_ENABLED = True
# Set to True to use Focal Loss instead of weighted CrossEntropyLoss
FOCAL_LOSS_ENABLED = False
FOCAL_LOSS_GAMMA = 2.0 # Gamma parameter for Focal Loss

# --- Evaluation Settings ---
# Set to True to save predictions and gold labels for detailed analysis
SAVE_PREDICTIONS = True

# --- Random Seed for Reproducibility ---
RANDOM_SEED = 42