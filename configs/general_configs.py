import re
import pathlib
import tensorflow as tf

DEBUG_LEVEL = 0
PROFILE = False

# PATHS
# - TRAIN -
TRAIN_DIR = pathlib.Path('./data/train/Fluo-N2DH-SIM+')
# TRAIN_DIR = pathlib.Path('./data/train/Fluo-N2DH-GOWT1')
# TRAIN_DIR = pathlib.Path('/media/rrtammyfs/labDatabase/CellTrackingChallenge/BGUSIM/Fluo-N2DH-BGUSIM/Train')

# - TEST -
TEST_DIR = pathlib.Path('/media/rrtammyfs/labDatabase/CellTrackingChallenge/Training/Silver_GT/Fluo-N2DH-GOWT1-ST')

# - TEST -
# INFERENCE_DIR = pathlib.Path('./data/BGU-IL-Fluo-N2DH-GOWT1')
# INFERENCE_DIR = pathlib.Path('./data/UNSW_AU_Fluo-N2DH-GOWT1')
INFERENCE_DIR = pathlib.Path('./data/nnUnet_Fluo-N2DH-SIM+')
# INFERENCE_DIR = pathlib.Path('./data/BGU_IL_Fluo-N2DH-GOWT1')

CHECKPOINT_DIR = pathlib.Path('./output/11_4_150_epochs (train)/checkpoints')

# - OUTPUT -
OUTPUT_DIR = pathlib.Path('/media/rrtammyfs/Users/sidorov/QANet/output')

TEMP_DIR = pathlib.Path('/media/rrtammyfs/Users/sidorov/QANet/temp')

CONFIGS_DIR_PATH = pathlib.Path('./configs')

RIBCAGE_CONFIGS_FILE_PATH = pathlib.Path('./configs/ribcage_configs.yml')

# - Regular expression to extract the images directories
METADATA_FILES_REGEX = re.compile(r'.+(?<=metadata)_[0-9]{2}.(?:pkl|pickle)')
# METADATA_FILES_REGEX = re.compile(r'.+(?<=metadata)_[0-9]{2}.pkl')

# CONSTANTS
EPSILON = 1e-7
ZERO_LOW_JACCARDS = False

# DATA
ORIGINAL_IMAGE_SHAPE = (718, 660)
ORIGINAL_IMAGE_MIN_VAL = 0.0
ORIGINAL_IMAGE_MAX_VAL = 255.0

# - Crops
IMAGE_SIZE = 256

# -- Object size
MIN_OBJECT_AREA = 150

# -- Shuffle
SHUFFLE = True

# PREPROCESSING CONFIGS
STANDARDIZE_IMAGE = False

# FILTERS CONFIGS
# - CLipped Adaptive Histogram Equalization (CLAHE)
# Enhances the contrast by equalizing the image intensity
# APPLY_CLAHE_FILTER = False
# CLAHE_CLIP_LIMIT = 2.0
# CLAHE_TILE_GRID_SIZE = (8, 8)

# AUGMENTATION CONFIGS
# - Morphological Transforms
EROSION = True
EROSION_SIZES = (10, 20, 30, 40, 50)

DILATION = True
DILATION_SIZES = (10, 20, 30, 40, 50)

OPENING = True
OPENING_SIZES = (1, 3, 5)

CLOSING = True
CLOSING_SIZES = (1, 3, 5)

# - Affine Transforms
AFFINE = True
SCALE_RANGE = (.01, .2)
SHEER_RANGE = (.05, .2)

# - Elastic Transforms
ELASTIC = True
SIGMA_RANGE = (1, 8)
ALPHA_RANGE = (50, 100)

# NN
ACTIVATION = 'swish'  # Swish()  # keras.layers.LeakyReLU()
ACTIVATION_ALPHA = 0.1
ACTIVATION_MAX_VALUE = None
ACTIVATION_THRESHOLD = 0.0

# > Training
OPTIMIZER = 'adam'
OPTIMIZER_RHO = 0.95
OPTIMIZER_BETA_1 = 0.9
OPTIMIZER_BETA_2 = 0.999
OPTIMIZER_MOMENTUM = 0.01

# - REGULARIZER
KERNEL_REGULARIZER_TYPE = 'l2'
KERNEL_REGULARIZER_L1 = 0.01
KERNEL_REGULARIZER_L2 = 0.01
KERNEL_REGULARIZER_FACTOR = 0.01
KERNEL_REGULARIZER_MODE = 'rows'
# OPTIMIZER = tf.keras.optimizers.Adam(
#     learning_rate=0.001,
#     beta_1=0.9,
#     beta_2=0.999,
#     epsilon=1e-07,
#     amsgrad=False,
#     name='Adam',
# )
# KERNEL_REGULARIZER = tf.keras.regularizers.L2(
#     l2=0.01
# )
LOSS = tf.keras.losses.MeanSquaredError()
METRICS = []

#  Variables
EPOCHS = 200
BATCH_SIZE = 256
VAL_BATCH_SIZE = 1
VALIDATION_PROPORTION = .2
LEARNING_RATE = 1e-3

# > CALLBACKS
# - Tensor Board
TENSOR_BOARD = True
TENSOR_BOARD_WRITE_IMAGES = True
TENSOR_BOARD_WRITE_STEPS_PER_SECOND = True
TENSOR_BOARD_UPDATE_FREQ = 'epoch'
TENSOR_BOARD_SCALARS_LOG_INTERVAL = 1
TENSOR_BOARD_IMAGES_LOG_INTERVAL = 1

# -> Scatter Plot
# PLOT_OUTLIERS = False
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

# -> Launch Tensor Board
TENSOR_BOARD_LAUNCH = True

# - Early Stopping
EARLY_STOPPING = False
EARLY_STOPPING_MONITOR = 'val_loss'
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.001
EARLY_STOPPING_MODE = 'auto'
EARLY_STOPPING_RESTORE_BEST_WEIGHTS = True
EARLY_STOPPING_VERBOSE = 1

# - Terminate on NaN
TERMINATE_ON_NAN = True

# - LR Reduce
REDUCE_LR_ON_PLATEAU = True
REDUCE_LR_ON_PLATEAU_MONITOR = 'val_loss'
REDUCE_LR_ON_PLATEAU_FACTOR = 0.5
REDUCE_LR_ON_PLATEAU_PATIENCE = 5
REDUCE_LR_ON_PLATEAU_MIN_DELTA = 0.01
REDUCE_LR_ON_PLATEAU_COOLDOWN = 0
REDUCE_LR_ON_PLATEAU_MIN_LR = 1e-7
REDUCE_LR_ON_PLATEAU_MODE = 'auto'
REDUCE_LR_ON_PLATEAU_VERBOSE = 1

# - Model Checkpoint
MODEL_CHECKPOINT = True
MODEL_CHECKPOINT_FILE_TEMPLATE = 'checkpoints/cp-{epoch:04d}.ckpt'  # <- may be used in case we want to save all teh check points, and not only the best
MODEL_CHECKPOINT_FILE_BEST_MODEL_TEMPLATE = 'checkpoints/best_model.ckpt'  # <- overwrites the second best model weights in case MODEL_CHECKPOINT_SAVE_BEST_ONLY = True
MODEL_CHECKPOINT_MONITOR = 'val_loss'
MODEL_CHECKPOINT_VERBOSE = 1
MODEL_CHECKPOINT_SAVE_BEST_ONLY = True
MODEL_CHECKPOINT_MODE = 'auto'
MODEL_CHECKPOINT_SAVE_WEIGHTS_ONLY = True
MODEL_CHECKPOINT_SAVE_FREQ = 'epoch'
