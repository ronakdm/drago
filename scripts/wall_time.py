"""
Train model for a particular objective and optimizer on evry hyperparameter setting.
"""

import time
import datetime
import sys
import argparse
import pickle
import os
import sys

sys.path.extend([".", ".."])
from src.utils.io import var_to_str, get_path

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
    "--seeds", type=str, required=True, choices=["0", "1", "2", "3", "5"]
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
seed_choices = {"0": SEEDS0, "1": SEEDS1, "2": SEEDS2, "3": SEEDS3, "5": SEEDS5}
model_cfg = {
    "objective": args.objective,
    "l2_reg": l2_reg,
    "loss": args.loss,
    "n_class": 6 if "emotion" in dataset else None,
    "sm_coef": sm_coefs[args.sm_coef],
    "smoothing": args.smoothing
}
hyperparam = bool(args.use_hyperparam)

# get best lr
lr = 0

# optim_cfg = {
#     "optimizer": args.optimizer,
#     "lr": lr,
#     "epoch_len": args.epoch_len,
#     "sm_coef": sm_coef,
#     "smoothing": args.smoothing
# }
n_epochs = args.n_epochs
save_iters = bool(args.save_iters)
redo = bool(args.redo)

# optim_cfgs = dict_to_list(optim_cfg)
seeds = seed_choices[args.seeds]
config = {
    "dataset": dataset,
    "model_cfg": model_cfg,
    "optimizer": args.optimizer,
    "seeds": seeds,
    "n_epochs": n_epochs,
    "epoch_len": args.epoch_len,
}

# Display.
print("-----------------------------------------------------------------")
print("-------------------------- WALL TIME ----------------------------")
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
path = get_path([dataset, var_to_str(model_cfg), args.optimizer])
with open(os.path.join(path, "best_cfg.p"), "rb") as f:
    optim = pickle.load(f)
worker(optim)
toc = time.time()
print(f"Time:         {format_time(toc-tic)}.")
