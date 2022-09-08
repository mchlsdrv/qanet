import pathlib
import torch.nn as nn

__author__ = 'sidorov@post.bgu.ac.il'

# > CHECK POINTS
CHECKPOINT_FILE_BEST_MODEL = pathlib.Path('/home/sidorov/Projects/QANetV2/qanet/output/train/pytorch_2022-08-08_17-28-30/checkpoints/best_val_loss_chkpt.pth.tar')

# TRAINING
# - Loss
# LOSS = nn.MSELoss()
LOSS = nn.L1Loss()
