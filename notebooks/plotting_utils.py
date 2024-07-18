import pickle
import os
import torch
import numpy as np

import sys
sys.path.extend(["..", "."])
from src.data import load_dataset
from src.utils import var_to_str, get_path, load_results, compute_average_train_loss, compute_time_loss, compute_loss_against_oracle_evals, get_suboptimality

short_to_filename = {
    "drago_1": "drago",
    "drago_16": "drago_auto",
    "drago_block": "drago_batch",
    "lsvrg": "slsvrg",
    "sgd": "sgd"
}

def plot_iterations(
    ax,
    dataset,
    model_cfg,
    plot_cfg,
    seeds,
    use_lbfgs=True,
    out_path="../results/",
    verbose=False,
    n_points=16,
    markersize=8,
    n_epochs=None,
    limit=None
):
    filename = short_to_filename[plot_cfg["optimizer"]]  # "code" name (e.g. "lsvrg")
    label = plot_cfg["label"]  # display name
    color = plot_cfg["color"]
    linestyle = plot_cfg["linestyle"]


    X_train = load_dataset(dataset, data_path="../data/")[0]
    n = len(X_train)
    d = X_train.shape[1]

    path = get_path([dataset, var_to_str(model_cfg), filename], out_path=out_path)
    if model_cfg["objective"] in plot_cfg:
        optim_cfg = {
            "optimizer": plot_cfg["optimizer"],
            "lr": plot_cfg[model_cfg["objective"]]["lr"],
            "epoch_len": plot_cfg[model_cfg["objective"]]["epoch_len"],
            "sm_coef": plot_cfg[model_cfg["objective"]]["sm_coef"],
        }
        avg_train_loss = compute_average_train_loss(
            dataset, model_cfg, optim_cfg, seeds, out_path=out_path
        )
        epoch_len = optim_cfg["epoch_len"]
        x, avg_train_loss = compute_loss_against_oracle_evals(dataset, model_cfg, optim_cfg, seeds, out_path=out_path)
    else:
        df = pickle.load(open(os.path.join(path, "best_traj.p"), "rb"))
        opt = pickle.load(open(os.path.join(path, "best_cfg.p"), "rb"))
        if verbose:
            print(f"{filename} best config:", opt)
        avg_train_loss = torch.tensor(df["average_train_loss"])
        epoch_len = opt["epoch_len"]
        x, avg_train_loss = compute_loss_against_oracle_evals(dataset, model_cfg, opt, seeds, out_path=out_path)
    subopt = get_suboptimality(
        dataset, model_cfg, avg_train_loss, use_lbfgs=use_lbfgs, out_path=out_path
    )
    # downsample = torch.sum(idx).item() // n_points
    if limit:
        subopt = subopt[x <= limit]
        x = x[x <= limit]

    idx = np.arange(len(subopt))

    if len(idx) > n_points:
        downsample = len(idx) // n_points
        x = x[::downsample]
        subopt = subopt[::downsample]
    ax.plot(
        x,
        subopt,
        color=color,
        label=label,
        linestyle=linestyle,
        marker=plot_cfg["marker"],
        markersize=markersize,
    )

def plot_runtime(
    ax,
    dataset,
    model_cfg,
    plot_cfg,
    seeds,
    use_lbfgs=True,
    out_path="../results/",
    verbose=False,
    n_points=16,
    markersize=8,
    n_epochs=None,
    limit=None
):
    filename = short_to_filename[plot_cfg["optimizer"]]  # "code" name (e.g. "lsvrg")
    label = plot_cfg["label"]  # display name
    color = plot_cfg["color"]
    linestyle = plot_cfg["linestyle"]


    X_train = load_dataset(dataset, data_path="../data/")[0]
    n = len(X_train)
    d = X_train.shape[1]

    path = get_path([dataset, var_to_str(model_cfg), filename], out_path=out_path)
    if model_cfg["objective"] in plot_cfg:
        optim_cfg = {
            "optimizer": plot_cfg["optimizer"],
            "lr": plot_cfg[model_cfg["objective"]]["lr"],
            "epoch_len": plot_cfg[model_cfg["objective"]]["epoch_len"],
            "sm_coef": plot_cfg[model_cfg["objective"]]["sm_coef"],
        }
        avg_train_loss = compute_average_train_loss(
            dataset, model_cfg, optim_cfg, seeds, out_path=out_path
        )
        epoch_len = optim_cfg["epoch_len"]
        x, avg_train_loss = compute_time_loss(dataset, model_cfg, optim_cfg, seeds, out_path=out_path)
    else:
        df = pickle.load(open(os.path.join(path, "best_traj.p"), "rb"))
        opt = pickle.load(open(os.path.join(path, "best_cfg.p"), "rb"))
        if verbose:
            print(f"{filename} best config:", opt)
        avg_train_loss = torch.tensor(df["average_train_loss"])
        epoch_len = opt["epoch_len"]
        x, avg_train_loss = compute_time_loss(dataset, model_cfg, opt, seeds, out_path=out_path)
    subopt = get_suboptimality(
        dataset, model_cfg, avg_train_loss, use_lbfgs=use_lbfgs, out_path=out_path
    )
    # downsample = torch.sum(idx).item() // n_points
    if limit:
        subopt = subopt[x <= limit]
        x = x[x <= limit]

    idx = np.arange(len(subopt))

    if len(idx) > n_points:
        downsample = len(idx) // n_points
        x = x[::downsample]
        subopt = subopt[::downsample]
    ax.plot(
        # x[idx][::downsample],
        # subopt[idx][::downsample],
        x,
        subopt,
        color=color,
        label=label,
        linestyle=linestyle,
        marker=plot_cfg["marker"],
        markersize=markersize,
    )
    # print(f"{dataset}:{model_cfg['objective']}:{filename}:{subopt[idx][::downsample][-1]}")