"""
Train model for a particular objective and optimizer on evry hyperparameter setting.
"""

import time
import datetime
from joblib import Parallel, delayed
import sys
import argparse

# Create parser.
sys.path.append(".")
from src.utils.hyperparams import HYPERPARAM_LR
from src.utils.config import (
    L2_REG,
    L2_REG_NONE,
    L2_REG_XSMALL,
    L2_REG_SMALL,
    L2_REG_LARGE,
    L2_REG_XLARGE,
    SM_XSMALL,
    SM_SMALL,
    SM_MEDIUM,
    SM_LARGE,
    LRS,
    SEEDS0,
    SEEDS1,
    SEEDS2,
    SEEDS3,
    SEEDS5,
    N_EPOCHS,
)
from src.utils.training import (
    OptimizationError,
    compute_training_curve,
    format_time,
    find_best_optim_cfg,
    FAIL_CODE,
)
from src.utils.io import dict_to_list

# Parse command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    required=True,
    # choices=[
    #     "yacht",
    #     "energy",
    #     "simulated",
    #     "concrete",
    #     "iwildcam_std",
    #     "emotion",
    #     "kin8nm",
    #     "naval",
    #     "power",
    #     "acsincome",
    #     "diabetes",
    #     "amazon",
    # ],
)
parser.add_argument(
    "--objective",
    type=str,
    required=True,
    choices=[
        "extremile",
        "superquantile",
        "esrm",
        "erm",
        "extremile_lite",
        "superquantile_lite",
        "esrm_lite",
        "extremile_hard",
        "superquantile_hard",
        "esrm_hard",
        "chi2"
    ],
)
parser.add_argument(
    "--optimizer",
    type=str,
    required=True,
)
parser.add_argument(
    "--loss",
    type=str,
    default="squared_error",
)
parser.add_argument(
    "--n_epochs",
    type=int,
    default=N_EPOCHS,
)
parser.add_argument(
    "--epoch_len",
    type=int,
    default=None,
)
parser.add_argument(
    "--l2_reg", type=str, required=True, choices=["none", "xsmall", "small", "medium", "large"]
)
parser.add_argument("--device", type=str, default="0", choices=["0", "1", "2", "3"])
parser.add_argument("--smoothing", type=str, default="l2", choices=["l2", "neg_entropy"])
parser.add_argument(
    "--sm_coef", type=str, required=True, choices=["xsmall", "small", "medium", "large"]
)
parser.add_argument(
    "--seeds", type=str, required=True, choices=["0", "1", "2", "3", "4", "5"]
)
parser.add_argument("--use_hyperparam", type=int, default=0)
parser.add_argument("--parallel", type=int, default=1)
parser.add_argument("--redo", type=int, default=0)
parser.add_argument("--save_iters", type=int, default=0)
parser.add_argument("--n_jobs", type=int, default=-2)
args = parser.parse_args()

# Configure for input to trainers.
dataset = args.dataset
l2_regs = {
    "none": L2_REG_NONE,
    "xsmall": L2_REG_XSMALL,
    "small": L2_REG_SMALL,
    "medium": L2_REG,
    "large": L2_REG_LARGE,
    "xlarge": L2_REG_XLARGE,
}
l2_reg = l2_regs[args.l2_reg]
sm_coefs = {
    "xsmall": SM_XSMALL,
    "small": SM_SMALL,
    "medium": SM_MEDIUM,
    "large": SM_LARGE,
}
sm_coef = sm_coefs[args.sm_coef]
seed_choices = {"0": SEEDS0, "1": SEEDS1, "2": SEEDS2, "3": [3, 4],  "4": [1, 2, 3, 4], "5": SEEDS5}
model_cfg = {
    "objective": args.objective,
    "l2_reg": l2_reg,
    "loss": args.loss,
    "n_class": None,
    "sm_coef": sm_coefs[args.sm_coef],
    "smoothing": args.smoothing
}
hyperparam = bool(args.use_hyperparam)
if hyperparam:
    lrs = HYPERPARAM_LR[args.optimizer][dataset][args.objective]
else:
    lrs = LRS
optim_cfg = {
    "optimizer": args.optimizer,
    "lr": lrs,
    "epoch_len": args.epoch_len,
    "sm_coef": sm_coef,
    "smoothing": args.smoothing
}
seeds = seed_choices[args.seeds]
n_epochs = args.n_epochs
parallel = bool(args.parallel)
save_iters = bool(args.save_iters)
redo = bool(args.redo)

optim_cfgs = dict_to_list(optim_cfg)

config = {
    "dataset": dataset,
    "model_cfg": model_cfg,
    "optim_cfg": optim_cfg,
    "parallel": parallel,
    "seeds": seeds,
    "n_epochs": n_epochs,
    "epoch_len": args.epoch_len,
}

# Display.
print("-----------------------------------------------------------------")
for key in config:
    print(f"{key}:" + " " * (16 - len(key)), config[key])
print(f"Start:" + " " * 11, {str(datetime.datetime.now())})
print("-----------------------------------------------------------------")


# Run optimization.
def worker(optim):
    name, lr = optim["optimizer"], optim["lr"]
    diverged = False
    for seed in seeds:
        code = compute_training_curve(
            dataset,
            model_cfg,
            optim,
            seed,
            n_epochs,
            device=args.device,
            save_iters=save_iters,
            redo=redo,
        )
        if code == FAIL_CODE:
            diverged = True
    if diverged:
        print(f"Optimizer '{name}' diverged at learning rate {lr}!")


tic = time.time()
if parallel:
    Parallel(n_jobs=args.n_jobs)(delayed(worker)(optim) for optim in optim_cfgs)
else:
    for optim in optim_cfgs:
        worker(optim)
toc = time.time()
print(f"Time:         {format_time(toc-tic)}.")

# Save best configuration.
find_best_optim_cfg(dataset, model_cfg, optim_cfgs, seeds)