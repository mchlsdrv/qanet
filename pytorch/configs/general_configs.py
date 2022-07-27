import pathlib

__author__ = 'sidorov@post.bgu.ac.il'

import torch.nn as nn

DEBUG_LEVEL = 0
PROFILE = False

# PATHS
# - TRAIN -
TRAIN_DIR = pathlib.Path('../../data/train/Fluo-N2DH-SIM+')
# TRAIN_DIR = pathlib.Path('./data/train/Fluo-N2DH-GOWT1')

# - TEST -
TEST_GT_DIR = pathlib.Path('../../data/test/Fluo-N2DH-GOWT1/GT')  # <= Gold (experts) standard test
TEST_ST_DIR = pathlib.Path('../../data/test/Fluo-N2DH-GOWT1/ST')  # <= Silver (best algorithm) standard test

# - TEST -
# INFERENCE_DIR = pathlib.Path('../../data/inference/BGU-IL-Fluo-N2DH-GOWT1')
INFERENCE_DIR = pathlib.Path('./data/inference/UNSW-AU-Fluo-N2DH-GOWT1')
# INFERENCE_DIR = pathlib.Path('./data/inference/UNSW-AU-Fluo-N2DH-SIM+')
# INFERENCE_DIR = pathlib.Path('./data/inference/DKFZ-GE-Fluo-N2DH-SIM+')
# INFERENCE_DIR = pathlib.Path('../../data/inference/BGU-IL-Fluo-N2DH-GOWT1')
# INFERENCE_DIR = pathlib.Path('../../data/inference/BGU-IL-Fluo-N2DH-GOWT1(1)')

CHECKPOINT_DIR = pathlib.Path('./output/11_4_150_epochs (train)/checkpoints')

# - OUTPUT -
OUTPUT_DIR = pathlib.Path('/media/rrtammyfs/Users/sidorov/QANet/output')

TEMP_DIR = pathlib.Path('./temp')
# TEMP_DIR = pathlib.Path('/media/rrtammyfs/Users/sidorov/QANet/temp')

CONFIGS_DIR = pathlib.Path('./configs')

MODEL_CONFIGS_FILE = CONFIGS_DIR / 'ribcage_configs.yml'


# CONSTANTS
EPSILON = 1e-7

# DATA
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
LOSS = nn.MSELoss()

LOAD_MODEL = False
CHECKPOINT_FILE = pathlib.Path('./chkpt/checkpoint.pth.tar')
EPOCHS = 10
BATCH_SIZE = 32
PIN_MEMORY = True
NUM_WORKERS = 2

EROSION_SIZES = [1, 3, 5, 7, 11]
DILATION_SIZES = [1, 3, 5, 7, 11]
COARSE_DROPOUT_MAX_HOLES = 8
COARSE_DROPOUT_MAX_HEIGHT = 100
COARSE_DROPOUT_MAX_WIDTH = 100
COARSE_DROPOUT_FILL_VALUE = 0.0
CLAHE_CLIP_LIMIT = 2
CLAHE_TILE_GRID_SIZE = 8
# - OPTIMIZER
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
# > Parameters
OPTIMIZER_MAXIMIZE = False
OPTIMIZER_EPS = 1e-8
OPTIMIZER_FOREACH = None

# - REGULARIZER
KERNEL_REGULARIZER_TYPE = 'l2'
KERNEL_REGULARIZER_L1 = 0.01
KERNEL_REGULARIZER_L2 = 0.01
KERNEL_REGULARIZER_FACTOR = 0.01
KERNEL_REGULARIZER_MODE = 'rows'

FALSE_POSITIVES_WEIGHT = .9
FALSE_NEGATIVES_WEIGHT = 1.2

IOU_THRESHOLD = 0.3
TRAIN_SCORE_THRESHOLD = 0.8
INFERENCE_SCORE_THRESHOLD = 0.85
N_TOP_PREDS = 100

#  Variables
VALIDATION_BATCH_SIZE = 10
VAL_PROP = .2
N_LOGS = 5

# > CALLBACKS
# - Early Stopping
EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10

# - LR Reduce
REDUCE_LR_ON_PLATEAU = True
REDUCE_LR_ON_PLATEAU_PATIENCE = 5
REDUCE_LR_ON_PLATEAU_FACTOR = 0.5
REDUCE_LR_ON_PLATEAU_MIN = 1e-8
