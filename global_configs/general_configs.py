import pathlib

__author__ = 'sidorov@post.bgu.ac.il'

# CONFIGS_DIR = pathlib.Path('./global_configs')

HYPER_PARAMS_FILE = pathlib.Path('./train_configs/hyper_parameters.yml')

# ========= DATA GEN MODULE =========
N_SAMPLES = 10000
MIN_J = 0.01
MAX_J = 0.99
MAX_J_TRIES = 1
MIN_CELL_PIXELS = 1000

# ========= MAIN MODULE =========
# TOGGLES
DEBUG_LEVEL = 0
PROFILE = False

# CONSTANTS
EPSILON = 1e-9

# - Scatter Plot
N_OUTLIERS = 5
OUTLIER_TH = .7

# - Hit Rate vs Absolute Error Tolerance Plot
HR_AET_FIGSIZE = (15, 15)
HR_AET_PERCENTAGE = 80

PLOT_OUTLIERS = True
PLOT_TRAIN_DATA_BATCHES = False
PLOT_VALIDATION_DATA_BATCHES = True
# LOSS_DELTA_TH = 0.01
# LAUNCH_TB = False
# METRICS = []

# VAL_BATCH_SIZE = 10
# REDUCE_LR_ON_PLATEAU_MIN_LR = 1e-9
# REDUCE_LR_ON_PLATEAU_VERBOSE = 1
