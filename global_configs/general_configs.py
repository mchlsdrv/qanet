import pathlib

__author__ = 'sidorov@post.bgu.ac.il'

# ========= DATA GEN MODULE =========
N_SAMPLES = 10000
MIN_J = 0.01
MAX_J = 0.99
MAX_J_TRIES = 1
MIN_CELL_PIXELS = 1000

# PATHS
# > DATA
# - DATA ROOT -
DATA_ROOT_DIR = pathlib.Path('../data')

# - TRAIN -
TRAIN_ROOT_DIR = DATA_ROOT_DIR / f'train'
TRAIN_DATA_NAME = 'Fluo-N2DH-SIM+'
TRAIN_DATA_DIR = TRAIN_ROOT_DIR / f'{TRAIN_DATA_NAME}'
TEMP_TRAIN_DATA_FILE = TRAIN_DATA_DIR / f'temp/{TRAIN_DATA_NAME}/clean_data.npy'

# - TEST -
TEST_ROOT_DIR = DATA_ROOT_DIR / f'test'
TEST_DATA_NAME = 'Fluo-N2DH-GOWT1-ST'
TEST_DATA_DIR = TEST_ROOT_DIR / f'{TEST_DATA_NAME}'
TEMP_TEST_DATA_FILE = TEST_DATA_DIR / f'temp/{TEST_DATA_NAME}/clean_data.npy'


# - INFERENCE -
INFERENCE_INPUT_DATA_DIR = 'UNSW-AU-Fluo-N2DH-GOWT1'
# INFERENCE_INPUT_DATA_DIR = 'BGU-IL-Fluo-N2DH-GOWT1'
# INFERENCE_INPUT_DATA_DIR = 'UNSW-AU-Fluo-N2DH-GOWT1'
# INFERENCE_INPUT_DATA_DIR = 'UNSW-AU-Fluo-N2DH-SIM+'
# INFERENCE_INPUT_DATA_DIR = 'DKFZ-GE-Fluo-N2DH-SIM+'
# INFERENCE_INPUT_DATA_DIR = 'BGU-IL-Fluo-N2DH-GOWT1'
# INFERENCE_INPUT_DATA_DIR = 'BGU-IL-Fluo-N2DH-GOWT1(1)'
INFERENCE_DATA_DIR = DATA_ROOT_DIR / f'inference/{INFERENCE_INPUT_DATA_DIR}'

# - OUTPUT -
OUTPUT_DIR = pathlib.Path('./output')

CONFIGS_DIR = pathlib.Path('./global_configs')

MODEL_CONFIGS_FILE = CONFIGS_DIR / 'ribcage_configs.yml'

# > CHECK POINTS
# -*- PyTorch
# TR_CHECKPOINT_FILE_BEST_MODEL = pathlib.Path('/home/sidorov/Projects/QANetV2/qanet/output/train/pytorch_2022-08-08_17-28-30/checkpoints/best_val_loss_chkpt.pth.tar')

# -*-  TensorFlow
# TF_CHECKPOINT_DIR = pathlib.Path('/home/sidorov/Projects/QANetV2/qanet/output/train/tensor_flow_2022-08-04_14-37-43/checkpoints')

# > STRINGS
SEG_DIR_POSTFIX = 'GT'
IMAGE_PREFIX = 't0'
SEG_PREFIX = 'man_seg0'
SEG_SUB_DIR = 'SEG'

# ========= MAIN MODULE =========
# TOGGLES
DEBUG_LEVEL = 0
PROFILE = False

# CONSTANTS
EPSILON = 1e-7

IMAGE_HEIGHT = 419
IMAGE_WIDTH = 419
IN_CHANNELS = 1
OUT_CHANNELS = 1

# SHUFFLE
SHUFFLE = True

# TRAINING
LOAD_MODEL = True
EPOCHS = 1000
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 10
TEST_BATCH_SIZE = 5
INF_BATCH_SIZE = 5
VAL_PROP = .2

NUM_TRAIN_WORKERS = 4
NUM_VAL_WORKERS = 4
PIN_MEMORY = True

# - Scatter Plot
N_OUTLIERS = 5
OUTLIER_TH = .7

PLOT_OUTLIERS = True
PLOT_TRAIN_DATA_BATCHES = False
PLOT_VALIDATION_DATA_BATCHES = True
LOSS_DELTA_TH = 0.01

# OPTIMIZER
# > Arguments
OPTIMIZER = 'adam'
OPTIMIZER_LR = 1e-4
OPTIMIZER_LR_DECAY = 0.001
OPTIMIZER_BETA_1 = 0.9
OPTIMIZER_BETA_2 = 0.999
OPTIMIZER_WEIGHT_DECAY = 0.01
OPTIMIZER_MOMENTUM = 0.1
OPTIMIZER_DAMPENING = 0.01
OPTIMIZER_MOMENTUM_DECAY = 0.01
OPTIMIZER_RHO = 0.9
OPTIMIZER_AMSGRAD = False

# Parameters
OPTIMIZER_MAXIMIZE = False
OPTIMIZER_EPS = 1e-8
OPTIMIZER_FOREACH = None

ACTIVATION = 'relu'
# ACTIVATION = 'swish'
ACTIVATION_RELU_MAX_VALUE = None
ACTIVATION_RELU_NEGATIVE_SLOPE = 0.0
ACTIVATION_RELU_THRESHOLD = 0.0
ACTIVATION_LEAKY_RELU_ALPHA = 0.1

# > Training
# - REGULARIZER
KERNEL_REGULARIZER_TYPE = 'l2'
KERNEL_REGULARIZER_L1 = 0.01
KERNEL_REGULARIZER_L2 = 0.01
KERNEL_REGULARIZER_FACTOR = 0.01
KERNEL_REGULARIZER_MODE = 'rows'
METRICS = []

DROP_BLOCK = True
DROP_BLOCK_KEEP_PROB = 0.4
DROP_BLOCK_BLOCK_SIZE = 20

# CALLBACKS
MIN_IMPROVEMENT_DELTA = 0.00001

# - Early Stopping
EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_MIN_DELTA = 0.0001
EARLY_STOPPING_MODE = 'auto'
EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True
EARLY_STOPPING_VERBOSE = 1

# - LR Reduction Scheduler
LR_REDUCTION_SCHEDULER = False
LR_REDUCTION_SCHEDULER_PATIENCE = [150, 300]
LR_REDUCTION_SCHEDULER_FACTOR = 0.5
LR_REDUCTION_SCHEDULER_MIN = 1e-9

# - LR Reduce on Plateau
REDUCE_LR_ON_PLATEAU = False
REDUCE_LR_ON_PLATEAU_PATIENCE = 10
REDUCE_LR_ON_PLATEAU_FACTOR = 0.8
REDUCE_LR_ON_PLATEAU_MIN = 1e-9
REDUCE_LR_ON_PLATEAU_MONITOR = 'val_loss'
REDUCE_LR_ON_PLATEAU_MIN_DELTA = 0.01
REDUCE_LR_ON_PLATEAU_COOLDOWN = 0
REDUCE_LR_ON_PLATEAU_MIN_LR = 1e-7
REDUCE_LR_ON_PLATEAU_MODE = 'auto'
REDUCE_LR_ON_PLATEAU_VERBOSE = 1
