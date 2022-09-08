import pathlib
import tensorflow as tf

__author__ = 'sidorov@post.bgu.ac.il'

# -*-  TensorFlow
CHECKPOINT_DIR = pathlib.Path('/home/sidorov/Projects/QANetV2/qanet/output/train/tensor_flow_2022-08-04_14-37-43/checkpoints')

# TRAINING
# - Loss
LOSS = tf.keras.losses.MeanSquaredError()

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

# PROGRESS_LOG = True
PROGRESS_LOG = False
PROGRESS_LOG_INTERVAL = 5
SCATTER_PLOT_FIGSIZE = (25, 15)
SCATTER_PLOT_FONTSIZE = 40

# - Terminate on NaN
TERMINATE_ON_NAN = True

# - Checkpoint
CHECKPOINT = True
CHECKPOINT_FILE_TEMPLATE = 'checkpoints/cp-{epoch:04d}.ckpt'  # <- may be used in case we want to save all teh check points, and not only the best
CHECKPOINT_FILE_BEST_MODEL = 'checkpoints/best_model.ckpt'  # <- overwrites the second-best model weights in case MODEL_CHECKPOINT_SAVE_BEST_ONLY = True
CHECKPOINT_MONITOR = 'val_loss'
CHECKPOINT_VERBOSE = 1
CHECKPOINT_SAVE_BEST_ONLY = True
CHECKPOINT_MODE = 'auto'
CHECKPOINT_SAVE_WEIGHTS_ONLY = True
CHECKPOINT_SAVE_FREQ = 'epoch'
