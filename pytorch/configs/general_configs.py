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
CHECKPOINT_FILE = pathlib.Path('/media/rrtammyfs/Users/sidorov/QANet/output/2022-07-27_22-57-53/checkpoints/best_val_loss_chkpt.pth.tar')
EPOCHS = 10
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 10
TEST_BATCH_SIZE = 5
VAL_PROP = .2

PIN_MEMORY = True
NUM_WORKERS = 2

EROSION_SIZES = (5, 10, 20)
DILATION_SIZES = (5, 10, 20)
OPENING_SIZES = (1, 3, 5)
CLOSING_SIZES = (1, 3, 5)
# - Affine Transforms
AFFINE = True
SCALE_RANGE = (.01, .2)
SHEER_RANGE = (.05, .2)
# - Elastic Transforms
ELASTIC = True
SIGMA_RANGE = (1, 8)
ALPHA_RANGE = (50, 100)

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

#  Variables
# > CALLBACKS
# - Early Stopping
EARLY_STOPPING = False
EARLY_STOPPING_PATIENCE = 20

# - LR Reduce
REDUCE_LR_ON_PLATEAU = False
REDUCE_LR_ON_PLATEAU_PATIENCE = 20
REDUCE_LR_ON_PLATEAU_FACTOR = 0.5
REDUCE_LR_ON_PLATEAU_MIN = 1e-8
