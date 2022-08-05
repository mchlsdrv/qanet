import pathlib
import torch.nn as nn
import tensorflow as tf

__author__ = 'sidorov@post.bgu.ac.il'

# TOGGLES
DEFAULT_MODEL_LIB = 'pytorch'
DEBUG_LEVEL = 0
PROFILE = False

# CONSTANTS
EPSILON = 1e-7

# PATHS
# - INPUT
RAW_DATA_DIR = pathlib.Path('../data')

GEN_DATA_DIR = pathlib.Path('../data/gen_data/Fluo-N2DH-SIM+')

# - TRAIN -
TRAIN_DATA_DIR = RAW_DATA_DIR / 'train/Fluo-N2DH-SIM+'
TRAIN_DATA_FILE = GEN_DATA_DIR / 'train/3942_samples/data.npy'

# - TEST -
TEST_DATA_DIR = RAW_DATA_DIR / 'test/Fluo-N2DH-SIM+'
TEST_GT_DATA_FILE = TEST_DATA_DIR / 'Fluo-N2DH-GOWT1/GT/gen_data/gt_data.npy'  # <= Gold (experts) standard test
TEST_ST_DATA_FILE = TEST_DATA_DIR / 'Fluo-N2DH-GOWT1/ST/gen_data/st_data.npy'  # <= Gold (experts) standard test

# - INFERENCE -
INFERENCE_DATA_DIR = RAW_DATA_DIR / 'inference/UNSW-AU-Fluo-N2DH-GOWT1'
# INFERENCE_DATA_DIR = RAW_DATA_DIR / 'inference/BGU-IL-Fluo-N2DH-GOWT1'
# INFERENCE_DATA_DIR = RAW_DATA_DIR / 'inference/UNSW-AU-Fluo-N2DH-GOWT1'
# INFERENCE_DATA_DIR = RAW_DATA_DIR / 'inference/UNSW-AU-Fluo-N2DH-SIM+'
# INFERENCE_DATA_DIR = RAW_DATA_DIR / 'inference/DKFZ-GE-Fluo-N2DH-SIM+'
# INFERENCE_DATA_DIR = RAW_DATA_DIR / 'inference/BGU-IL-Fluo-N2DH-GOWT1'
# INFERENCE_DATA_DIR = RAW_DATA_DIR / 'inference/BGU-IL-Fluo-N2DH-GOWT1(1)'

# - OUTPUT -
OUTPUT_DIR = pathlib.Path('./output')

CONFIGS_DIR = pathlib.Path('./configs')

# PATHS
MODEL_CONFIGS_FILE = CONFIGS_DIR / 'ribcage_configs.yml'

# DATA
N_SAMPLES = 10000
SEG_DIR_POSTFIX = 'GT'
IMAGE_PREFIX = 't0'
SEG_PREFIX = 'man_seg0'
IMAGE_HEIGHT = 419
IMAGE_WIDTH = 419
IN_CHANNELS = 1
OUT_CHANNELS = 1

# SHUFFLE
SHUFFLE = True

# TRAINING
# - Loss
TORCH_LOSS = nn.MSELoss()
TF_LOSS = tf.keras.losses.MeanSquaredError()

LOAD_MODEL = False
EPOCHS = 10
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 10
TEST_BATCH_SIZE = 5
INF_BATCH_SIZE = 5
VAL_PROP = .2

PIN_MEMORY = True
NUM_WORKERS = 2

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

ACTIVATION = 'swish'
ACTIVATION_RELU_MAX_VALUE = None
ACTIVATION_RELU_NEGATIVE_SLOPE = None
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
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_MIN_DELTA = 0.001
EARLY_STOPPING_MODE = 'auto'
EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True
EARLY_STOPPING_VERBOSE = 1

# - LR Reduce
REDUCE_LR_ON_PLATEAU = True
REDUCE_LR_ON_PLATEAU_PATIENCE = 10
REDUCE_LR_ON_PLATEAU_FACTOR = 0.5
REDUCE_LR_ON_PLATEAU_MIN = 1e-9
REDUCE_LR_ON_PLATEAU_MONITOR = 'val_loss'
REDUCE_LR_ON_PLATEAU_MIN_DELTA = 0.01
REDUCE_LR_ON_PLATEAU_COOLDOWN = 0
REDUCE_LR_ON_PLATEAU_MIN_LR = 1e-7
REDUCE_LR_ON_PLATEAU_MODE = 'auto'
REDUCE_LR_ON_PLATEAU_VERBOSE = 1

# - Tensor Board
TENSOR_BOARD_LAUNCH = True
TENSOR_BOARD = True
TENSOR_BOARD_WRITE_IMAGES = True
TENSOR_BOARD_WRITE_STEPS_PER_SECOND = True
TENSOR_BOARD_UPDATE_FREQ = 'epoch'
TENSOR_BOARD_SCALARS_LOG_INTERVAL = 1
TENSOR_BOARD_IMAGES_LOG_INTERVAL = 1

# - Scatter Plot
PLOT_OUTLIERS = True
PLOT_TRAIN_DATA_BATCHES = False
PLOT_VALIDATION_DATA_BATCHES = True
LOSS_DELTA_TH = 0.01
N_OUTLIERS = 5
OUTLIER_TH = .7
PROGRESS_LOG = True
PROGRESS_LOG_INTERVAL = 5
SCATTER_PLOT_FIGSIZE = (25, 15)
SCATTER_PLOT_FONTSIZE = 40

# - Terminate on NaN
TERMINATE_ON_NAN = True

# - Checkpoint
CHECKPOINT = True
TF_CHECKPOINT_FILE_TEMPLATE = 'checkpoints/cp-{epoch:04d}.ckpt'  # <- may be used in case we want to save all teh check points, and not only the best
TR_CHECKPOINT_FILE_BEST_MODEL = pathlib.Path('/media/rrtammyfs/Users/sidorov/QANet/output/pytorch_2022-07-27_22-57-53/checkpoints/best_val_loss_chkpt.pth.tar')
TF_CHECKPOINT_DIR = pathlib.Path('/media/rrtammyfs/Users/sidorov/QANet/output/tensor_flow_2022-07-27_22-57-53/checkpoints')
TF_CHECKPOINT_FILE_BEST_MODEL = 'checkpoints/best_model.ckpt'  # <- overwrites the second-best model weights in case MODEL_CHECKPOINT_SAVE_BEST_ONLY = True
CHECKPOINT_MONITOR = 'val_loss'
CHECKPOINT_VERBOSE = 1
CHECKPOINT_SAVE_BEST_ONLY = True
CHECKPOINT_MODE = 'auto'
CHECKPOINT_SAVE_WEIGHTS_ONLY = True
CHECKPOINT_SAVE_FREQ = 'epoch'
