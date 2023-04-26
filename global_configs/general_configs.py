import pathlib

__author__ = 'sidorov@post.bgu.ac.il'


HYPER_PARAMS_FILE = pathlib.Path('./train_configs/hyper_parameters.yml')

# ========= DATA GEN MODULE =========
N_SAMPLES = 10000
MIN_J = 0.01
MAX_J = 0.99
MAX_J_TRIES = 1
MIN_CELL_PIXELS = 1000

# ========= MAIN MODULE =========
# TOGGLES
# DEBUG = True
DEBUG = False
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

COLUMN_NAMES = ['image_file', 'gt_mask_file', 'pred_mask_file', 'seg_score']
