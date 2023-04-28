import pathlib

from configs.general_configs import (
    REDUCE_LR_ON_PLATEAU_MIN,
    REDUCE_LR_ON_PLATEAU,
    EARLY_STOPPING,
    EARLY_STOPPING_PATIENCE,
    REDUCE_LR_ON_PLATEAU_FACTOR,
    REDUCE_LR_ON_PLATEAU_PATIENCE,
    MIN_IMPROVEMENT_DELTA,
    LR_REDUCTION_SCHEDULER,
    LR_REDUCTION_SCHEDULER_PATIENCE,
    LR_REDUCTION_SCHEDULER_FACTOR,
    LR_REDUCTION_SCHEDULER_MIN
)
from utils.aux_funcs import save_checkpoint


# - CALLBACKS
# > Save Checkpoint on Improvement
def save_checkpoint_on_improvement(model, optimizer, output_dir: pathlib.Path):
    ckpt = dict(
        state_dict=model.state_dict(),
        optimizer=optimizer.state_dict(),
    )
    save_checkpoint(state=ckpt, filename=str(output_dir / f'best_val_loss_chkpt.pth.tar'))


# > Early Stopping
def early_stopping(no_improvement_epochs: int):
    stop_training = False

    if EARLY_STOPPING:
        if no_improvement_epochs >= EARLY_STOPPING_PATIENCE:
            print(f'<x> No improvement was recorded for {EARLY_STOPPING_PATIENCE} epochs - stopping the training!')
            stop_training = True
        else:
            if no_improvement_epochs > 0:
                print(f'<!> {no_improvement_epochs}/{EARLY_STOPPING_PATIENCE} of epochs without improvement recorded (i.e, stopping the training when counter reaches {EARLY_STOPPING_PATIENCE} consecutive epochs with improvements < {MIN_IMPROVEMENT_DELTA})!')
    return stop_training


# > LR Reduction on Plateau
def reduce_lr_on_plateau(no_improvement_epochs: int, optimizer):
    stop_training = False

    if REDUCE_LR_ON_PLATEAU:
        if no_improvement_epochs >= REDUCE_LR_ON_PLATEAU_PATIENCE:
            lr = optimizer.param_groups[0]['lr']
            new_lr = REDUCE_LR_ON_PLATEAU_FACTOR * lr
            if new_lr < REDUCE_LR_ON_PLATEAU_MIN:
                print(f'<x> The lr ({new_lr}) was reduced beyond its smallest possible value ({REDUCE_LR_ON_PLATEAU_MIN}) - stopping the training!')
                stop_training = True

            optimizer.param_groups[0]['lr'] = new_lr

            print(f'<!> No improvement was recorded for {REDUCE_LR_ON_PLATEAU_PATIENCE} epochs - reducing lr by factor {REDUCE_LR_ON_PLATEAU_FACTOR}, from {lr} -> {new_lr}!')
    return stop_training


def lr_reduction_scheduler(epoch: int, optimizer):
    stop_training = False

    if LR_REDUCTION_SCHEDULER:
        if epoch in LR_REDUCTION_SCHEDULER_PATIENCE:
            lr = optimizer.param_groups[0]['lr']
            new_lr = LR_REDUCTION_SCHEDULER_FACTOR * lr
            if new_lr < LR_REDUCTION_SCHEDULER_MIN:
                print(f'<x> The lr ({new_lr}) was reduced beyond its smallest possible value ({LR_REDUCTION_SCHEDULER_MIN}) - stopping the training!')
                stop_training = True

            optimizer.param_groups[0]['lr'] = new_lr

            print(f'<!> Reducing learning rate after epoch {epoch} > {LR_REDUCTION_SCHEDULER_PATIENCE} epochs - reducing lr by factor {LR_REDUCTION_SCHEDULER_FACTOR}, from {lr} -> {new_lr}!')

    return stop_training
