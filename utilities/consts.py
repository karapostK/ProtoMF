# --- Experiment Constants --- #
SINGLE_SEED = 38210573
SEED_LIST = [SINGLE_SEED, 9491758, 2931009]
NUM_SAMPLES = 100
# --- Training Constants --- #
MAX_PATIENCE = 10
# --- Evaluation Constants --- #
K_VALUES = [1, 3, 5, 10, 50]  # K value for the evaluation metrics
NEG_VAL = 99  # How many negative samples are considered during negative sampling
OPTIMIZING_METRIC = 'hit_ratio@10'  # Which metric will be used to assess during validation.
# --- Logger Constants --- #
WANDB_API_KEY = '<YOUR KEY HERE>'
PROJECT_NAME = 'protomf'
