import sys

sys.path.extend([".", ".."])
from src.baselines import StochasticSubgradientMethod, LSVRG
from src.drago import Drago
from src.objective import SpectralRiskMeasureObjective, DivergenceBallObjective, get_superquantile_weights

def get_optimizer(optim_cfg, objective, seed):
    name, lr, epoch_len, sm_coef = (
        optim_cfg["optimizer"],
        optim_cfg["lr"],
        optim_cfg["epoch_len"],
        optim_cfg["shift_cost"],
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
    elif name == "drago_batch":
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