import re
import pathlib
import tensorflow as tf


DEBUG_LEVEL = 0

# PROFILE = True
PROFILE = False

# PATHS
# - Path to the root data directory
ROOT_DIR = pathlib.Path('/media/rrtammyfs/labDatabase/CellTrackingChallenge/BGUSIM/Fluo-N2DH-BGUSIM/Train')

# - Path to the images directory
IMAGE_DIR = pathlib.Path('./data/train/imgs')

# The sub folder where the actual images are placed
IMAGE_SUB_DIR = None

# - Regular expression to extract the images directories
IMAGE_DIR_REGEX = re.compile(r'(?<![a-zA-Z_])[0-9]{2}(?![a-zA-Z_])')

# - Path to the segmentations directory
SEGMENTATION_DIR = pathlib.Path('./data/train/segs')

# The sub folder where the actual segmentations are placed
SEGMENTATION_SUB_DIR = 'SEG'

# - Regular expression to extract the segmentations directories
SEGMENTATION_DIR_REGEX = re.compile(r'(?<![a-z])[0-9]{2}_GT')

OUTPUT_DIR = pathlib.Path('./output')

TEMP_DIR = pathlib.Path('./temp')

CONFIGS_DIR_PATH = pathlib.Path('./configs')

RIBCAGE_CONFIGS_FILE_PATH = pathlib.Path('./configs/ribcage_configs.yml')

# CONSTANTS
EPSILON = 1e-7
ZERO_LOW_JACCARDS = True

# DATA
# - Crops
CROP_SIZE = 256
NON_EMPTY_CROPS = False
# NON_EMPTY_CROPS = True
MIN_OBJECT_AREA = 400
NON_EMPTY_CROP_THRESHOLD = 2000
MAX_EMPTY_CROPS = 3

NUMBER_INDEX_CHARACTERS = 3
SHUFFLE_CROPS = True

# PREPROCESSING CONFIGS
STANDARDIZE_IMAGE = False

# FILTERS CONFIGS
# - CLipped Adaptive Histogram Equalization (CLAHE)
# Enhances the contrast by equalizing the image intensity
APPLY_CLAHE_FILTER = True
CLAHE_CLIP_LIMIT = 30.0
CLAHE_TILE_GRID_SIZE = (10, 10)

# AUGMENTATION CONFIGS
# ROTATION = True
ROTATION = False

# - Morphological Transforms
EROSION = True
EROSION_SIZES = (1, 3, 5, 7, 11)

DILATION = True
DILATION_SIZES = (1, 3, 5, 7, 11)

OPENING = True
OPENING_SIZES = (1, 3, 5, 7, 11)

CLOSING = True
CLOSING_SIZES = (1, 3, 5, 7, 11)

# - Affine Transforms
AFFINE = True
SCALE_RANGE = (.01, .2)
SHEER_RANGE = (.05, .2)


# - Elastic Transforms
ELASTIC = True
SIGMA_RANGE = (1, 8)
ALPHA_RANGE = (50, 100)

# NN
# > Training
OPTIMIZER = tf.keras.optimizers.Adam
LOSS = tf.keras.losses.MeanSquaredError()
METRICS = ['acc']

#  Variables
EPOCHS = 10
BATCH_SIZE = 10
VAL_BATCH_SIZE = 1
VALIDATION_PROPORTION = .2
LEARNING_RATE = 1e-4

# > CALLBACKS
# - Tensor Board
TENSOR_BOARD = True
TENSOR_BOARD_WRITE_GRAPH = False
TENSOR_BOARD_WRITE_IMAGES = True
TENSOR_BOARD_WRITE_STEPS_PER_SECOND = True
TENSOR_BOARD_UPDATE_FREQ = 'epoch'

TENSOR_BOARD_LOG_INTERVAL = 1
TENSOR_BOARD_SCALARS_LOG_INTERVAL = 1
TENSOR_BOARD_IMAGES_LOG_INTERVAL = 25

# -> Scatter Plot
PLOT_SCATTER = True
SCATTER_PLOT_LOG_INTERVAL = 25
SCATTER_PLOT_FIGSIZE = (25, 15)
SCATTER_PLOT_FONTSIZE = 40

# -> Launch Tensor Board
TENSOR_BOARD_LAUNCH = True

# - Early Stopping
EARLY_STOPPING = True
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 50
EARLY_STOPPING_MIN_DELTA = 0
EARLY_STOPPING_MODE = 'auto'
EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True
EARLY_STOPPING_VERBOSE = 1

# - Terminate on NaN
TERMINATE_ON_NAN = True

# - LR Reduce
REDUCE_LR_ON_PLATEAU = True
REDUCE_LR_ON_PLATEAU_MONITOR = 'val_loss'
REDUCE_LR_ON_PLATEAU_FACTOR = 0.1
REDUCE_LR_ON_PLATEAU_PATIENCE = 100
REDUCE_LR_ON_PLATEAU_MIN_DELTA = 0.0001
REDUCE_LR_ON_PLATEAU_COOLDOWN = 0
REDUCE_LR_ON_PLATEAU_MIN_LR = 0.0
REDUCE_LR_ON_PLATEAU_MODE = 'auto'
REDUCE_LR_ON_PLATEAU_VERBOSE = 1

# - Model Checkpoint
MODEL_CHECKPOINT = True
MODEL_CHECKPOINT_FILE_TAMPLATE = 'checkpoints/cp-{epoch:04d}.ckpt'  # <- may be used in case we want to save all teh check points, and not only the best
MODEL_CHECKPOINT_FILE_BEST_MODEL_TAMPLATE = 'checkpoints/best_model.ckpt'  # <- overwrites the second best model weigths in case MODEL_CHECKPOINT_SAVE_BEST_ONLY = True
MODEL_CHECKPOINT_MONITOR = 'val_loss'
MODEL_CHECKPOINT_VERBOSE = 1
MODEL_CHECKPOINT_SAVE_BEST_ONLY = True
MODEL_CHECKPOINT_MODE = 'auto'
MODEL_CHECKPOINT_SAVE_WEIGHTS_ONLY = True
MODEL_CHECKPOINT_SAVE_FREQ = 'epoch'
