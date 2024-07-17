import torch
import numpy as np
import os
import pickle
import pandas as pd
from scipy.optimize import minimize
import inspect
import sys

sys.path.extend([".", ".."])
from src.baselines import StochasticSubgradientMethod, LSVRG
from src.drago import Drago
from src.objective import SpectralRiskMeasureObjective, DivergenceBallObjective, get_superquantile_weights

FAIL_CODE = -1

def get_optimizer(optim_cfg, objective, seed):
    name, lr, epoch_len, sm_coef = (
        optim_cfg["optimizer"],
        optim_cfg["lr"],
        optim_cfg["epoch_len"],
        optim_cfg["dual_reg"],
    )

    if name == "sgd":
        return StochasticSubgradientMethod(
            objective, lr=lr, seed=seed, epoch_len=epoch_len
        )
    if name == "lsvrg":
        return LSVRG(
                objective,
                lr=lr,
                smooth_coef=sm_coef,
                smoothing="l2",
                seed=seed,
                uniform=True,
                length_epoch=epoch_len,
            )
    elif name == "drago":
        return Drago(objective, lr=lr, epoch_len=epoch_len, block_size=1 if not ("block_size" in optim_cfg) else optim_cfg["block_size"], sm_coef=sm_coef)
    elif name == "drago_auto":
        return Drago(objective, lr=lr, epoch_len=epoch_len, block_size="auto", sm_coef=sm_coef)
    elif name == "drago_block":
        return Drago(objective, lr=lr, epoch_len=epoch_len, block_size=None, sm_coef=sm_coef)
    else:
        raise ValueError("Unrecognized optimizer!")
    
def get_objective(model_cfg, X, y, dataset=None, autodiff=False):
    name, l2_reg, loss, n_class, sm_coef = (
        model_cfg["objective"],
        model_cfg["l2_reg"],
        model_cfg["loss"],
        model_cfg["n_class"],
        model_cfg["shift_cost"],
    )
    if name == "chi2":
        return DivergenceBallObjective(
            X,
            y,
            1.0,
            l2_reg=l2_reg,
            loss=loss,
            n_class=n_class,
            risk_name=name,
            dataset=dataset,
            sm_coef=sm_coef,
            smoothing="l2",
            autodiff=autodiff
        )
    elif name == "cvar":
        weight_function = lambda n: get_superquantile_weights(n, 0.75)
        return SpectralRiskMeasureObjective(
            X,
            y,
            weight_function,
            l2_reg=l2_reg,
            loss=loss,
            n_class=n_class,
            risk_name=name,
            dataset=dataset,
            sm_coef=sm_coef,
            smoothing="l2",
            autodiff=autodiff
        )
    else:
        raise ValueError("Unrecognized objective!")
    
def get_min_loss(model_cfg, X_train, y_train):
    
    # compare to a reference loss from L-BFGS (second-order method)
    train_obj_ = get_objective(model_cfg, X_train, y_train)

    # Define function and Jacobian oracles.
    def fun(w):
        return train_obj_.get_batch_loss(torch.tensor(w, dtype=torch.float64)).item()

    def jac(w):
        return (
            train_obj_.get_batch_subgrad(
                torch.tensor(w, dtype=torch.float64, requires_grad=True)
            )
            .detach()
            .numpy()
        )

    # Run optimizer.
    d = train_obj_.d
    init = np.zeros((d,), dtype=np.float64)
    if model_cfg["n_class"]:
        init = np.zeros((model_cfg["n_class"] * d,), dtype=np.float64)
    else:
        init = np.zeros((d,), dtype=np.float64)
    output = minimize(fun, init, method="L-BFGS-B", jac=jac)
    return output.fun

def load_results(dataset, model_cfg, optim_cfg, seed, out_path="results/"):
    if "iwildcam" in dataset:
        model_cfg["n_class"] = 60
    if "emotion" in dataset:
        model_cfg["n_class"] = 6
    if "amazon" in dataset:
        model_cfg["n_class"] = 5
    path = get_path(
        [dataset, var_to_str(model_cfg), var_to_str(optim_cfg)], out_path=out_path
    )
    f = os.path.join(path, f"seed_{seed}.p")
    return pickle.load(open(f, "rb"))

def get_path(levels, out_path="results/"):
    path = out_path
    for item in levels:
        path = os.path.join(path, item + "/")
        if not os.path.exists(path):
            os.mkdir(path)
    return path

def var_to_str(var):
    translate_table = {ord(c): None for c in ",()[]"}
    translate_table.update({ord(" "): "_"})

    if type(var) == dict:
        sortedkeys = sorted(var.keys(), key=lambda x: x.lower())
        var_str = [
            key + "_" + var_to_str(var[key])
            for key in sortedkeys
            if var[key] is not None
        ]
        var_str = "_".join(var_str)
    elif inspect.isclass(var):
        raise NotImplementedError("Do not give classes as items in cfg inputs")
    elif type(var) in [list, set, frozenset, tuple]:
        value_list_str = [var_to_str(item) for item in var]
        var_str = "_".join(value_list_str)
    elif isinstance(var, float):
        var_str = "{0:1.2e}".format(var)
    elif isinstance(var, int):
        var_str = str(var)
    elif isinstance(var, str):
        var_str = var
    elif var is None:
        var_str = str(var)
    else:
        raise NotImplementedError
    return var_str

def compute_average_train_loss(
    dataset, model_cfg, optim_cfg, seeds, out_path="results/"
):
    total = 0.0
    for seed in seeds:
        results = load_results(dataset, model_cfg, optim_cfg, seed, out_path=out_path)
        if isinstance(results, int) and results == FAIL_CODE:
            return [torch.inf]
        total += torch.tensor(results["metrics"]["train_loss"])
    return total / len(seeds)

def compute_x_y(
    dataset, model_cfg, optim_cfg, seeds, out_path="results/"
):
    total = 0.0
    for seed in seeds:
        results = load_results(dataset, model_cfg, optim_cfg, seed, out_path=out_path)
        if isinstance(results, int) and results == FAIL_CODE:
            return [torch.inf]
        total += torch.tensor(results["metrics"]["train_loss"])
    y = total / len(seeds)
    x = results["metrics"]["oracle_evals"]
    return x, y

def get_seed_trajectories(dataset, model_cfg, optim_cfg, seeds, out_path="results/"):
    paths = {}
    for seed in seeds:
        results = load_results(dataset, model_cfg, optim_cfg, seed, out_path=out_path)
        if isinstance(results, int) and results == FAIL_CODE:
            raise ValueError("Loaded failed trajectory")
        paths[f"seed_{seed}_train"] = torch.tensor(results["metrics"]["train_loss"])
    return pd.DataFrame(paths)

def compute_time_loss(
    dataset, model_cfg, optim_cfg, seeds, out_path="results/"
):
    total = 0.0
    for seed in seeds:
        results = load_results(dataset, model_cfg, optim_cfg, seed, out_path=out_path)
        if isinstance(results, int) and results == FAIL_CODE:
            return [torch.inf]
        total += torch.tensor(results["metrics"]["train_loss"])
    y = total / len(seeds)
    eps = 1e-8
    x = np.cumsum(eps + results["metrics"]["elapsed"].to_numpy())
    return x, y

def get_suboptimality(
    dataset, model_cfg, train_loss, eps=1e-9, use_lbfgs=True, out_path="../results/"
):
    init_loss = train_loss[0]

    if use_lbfgs:
        path = get_path([dataset, var_to_str(model_cfg)], out_path=out_path)
        f = os.path.join(path, "lbfgs_min_loss.p")
        min_loss = pickle.load(open(f, "rb"))
    else:
        path = get_path([dataset, var_to_str(model_cfg), "lsvrg"], out_path=out_path)
        df = pickle.load(open(os.path.join(path, "best_traj.p"), "rb"))
        avg_train_loss = torch.tensor(df["average_train_loss"])
        min_loss1 = avg_train_loss[-1]
        path = get_path(
            [dataset, var_to_str(model_cfg), "lsvrg_uniform"], out_path=out_path
        )
        df = pickle.load(open(os.path.join(path, "best_traj.p"), "rb"))
        avg_train_loss = torch.tensor(df["average_train_loss"])
        min_loss2 = avg_train_loss[-1]
        min_loss = min(min_loss1, min_loss2)
    subopt = (train_loss - min_loss + eps) / (init_loss - min_loss)
    # return torch.log10(subopt) # use if not setting yscale to log.
    return subopt