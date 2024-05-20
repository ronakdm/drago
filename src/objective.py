import torch
import sys
import math

sys.path.extend([".", ".."])
from src.dual import get_smooth_weights, get_smooth_weights_sorted, chi2_divergence_oracle

def squared_error_loss(w, X, y):
    return 0.5 * (y - torch.matmul(X, w)) ** 2

# def squared_error_gradient(w, X, y):
#     return torch.matmul((torch.matmul(X, w) - y), X)

def squared_error_gradient(w, X, y):
    return (torch.matmul(X, w) - y)[:, None] * X


def binary_cross_entropy_loss(w, X, y):
    logits = torch.matmul(X, w)
    return torch.nn.functional.binary_cross_entropy_with_logits(
        logits, y, reduction="none"
    )

def binary_cross_entropy_gradient(w, X, y):
    logits = torch.matmul(X, w)
    p = 1. / (1. + torch.exp(-logits))
    return (p - y)[:, None] * X

def multinomial_cross_entropy_loss(w, X, y, n_class):
    W = w.view(-1, n_class)
    logits = torch.matmul(X, W)
    return torch.nn.functional.cross_entropy(logits, y, reduction="none")

def multinomial_cross_entropy_gradient(w, X, y, n_class):
    n = len(X)
    W = w.view(-1, n_class)
    logits = torch.matmul(X, W)
    p = torch.softmax(logits, dim=1)
    p[torch.arange(n), y] -= 1
    scores = torch.bmm(X[:, :, None], p[:, None, :])
    return scores.view(n, -1)


def get_loss(name, n_class=None):
    if name == "squared_error":
        return squared_error_loss
    elif name == "binary_cross_entropy":
        return binary_cross_entropy_loss
    elif name == "multinomial_cross_entropy":
        return lambda w, X, y: multinomial_cross_entropy_loss(w, X, y, n_class)
    else:
        raise ValueError(
            f"Unrecognized loss '{name}'! Options: ['squared_error', 'binary_cross_entropy', 'multinomial_cross_entropy']"
        )
    
def get_grad_batch(name, n_class=None):
    if name == "squared_error":
        return squared_error_gradient
    elif name == "binary_cross_entropy":
        return binary_cross_entropy_gradient
    elif name == "multinomial_cross_entropy":
        return lambda w, X, y: multinomial_cross_entropy_gradient(w, X, y, n_class)
    else:
        raise NotImplementedError
    
def get_superquantile_weights(n, q):
    weights = torch.zeros(n, dtype=torch.float64)
    idx = math.floor(n * q)
    frac = 1 - (n - idx - 1) / (n * (1 - q))
    if frac > 1e-12:
        weights[idx] = frac
        weights[(idx + 1) :] = 1 / (n * (1 - q))
    else:
        weights[idx:] = 1 / (n - idx)
    return weights

class SpectralRiskMeasureObjective:
    def __init__(
        self,
        X,
        y,
        weight_function,
        loss="squared_error",
        l2_reg=None,
        n_class=None,
        risk_name=None,
        dataset=None,
        sm_coef=1.0,
        smoothing=None,
        autodiff=False
    ):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.weight_function = weight_function
        self.loss = get_loss(loss, n_class=n_class)
        # self.grad = get_grad(loss, n_class=n_class)
        self.grad_batch = get_grad_batch(loss, n_class=n_class)
        self.loss_name = loss
        self.n_class = n_class
        self.l2_reg = l2_reg
        self.autodiff = autodiff

        # For logging.
        self.loss_name = loss
        self.risk_name = risk_name
        self.dataset = dataset

        self.sigmas = weight_function(self.n)
        # self.smooth_coef = self.n * sm_coef if smoothing == "l2" else sm_coef
        self.smooth_coef = sm_coef
        self.smoothing = smoothing
        # TODO: Change for other penalties.
        self.penalty = (
            0.5 * self.smooth_coef * torch.sum((self.sigmas - 1 / self.n) ** 2)
        )

    @torch.no_grad()
    def get_batch_loss(self, w, include_reg=True):
        sorted_losses = torch.sort(self.loss(w, self.X, self.y), stable=True)[0]
        n = self.n
        if self.smoothing:
            sm_sigmas = get_smooth_weights_sorted(
                sorted_losses, self.sigmas, self.smooth_coef, self.smoothing
            )
            # TODO: Change this for l2.
            risk = torch.dot(
                sm_sigmas, sorted_losses
            ) - 0.5 * self.smooth_coef * torch.sum((sm_sigmas - 1 / n) ** 2)
        else:
            risk = torch.dot(self.sigmas, sorted_losses)
        if self.l2_reg and include_reg:
            # risk += 0.5 * self.l2_reg * torch.norm(w) ** 2 / self.n
            risk += 0.5 * self.l2_reg * torch.norm(w) ** 2
        return risk
    
    def get_batch_subgrad(self, w, idx=None, include_reg=True):
        # if self.autodiff:
        #     return self.get_batch_subgrad_autodiff(w, idx=idx, include_reg=include_reg)
        # else:
        return self.get_batch_subgrad_oracle(w, idx=idx, include_reg=include_reg)

    @torch.no_grad()
    def get_batch_subgrad_oracle(self, w, idx=None, include_reg=True):
        if idx is not None:
            X, y = self.X[idx], self.y[idx]
            sigmas = self.weight_function(len(X))
        else:
            X, y = self.X, self.y
            sigmas = self.sigmas
        sorted_losses, perm = torch.sort(self.loss(w, X, y), stable=True)
        if self.smoothing:
            q = get_smooth_weights_sorted(
                sorted_losses, sigmas, self.smooth_coef, self.smoothing
            )
        else:
            q = sigmas
        g = torch.matmul(q, self.grad_batch(w, X, y)[perm])
        if self.l2_reg and include_reg:
            # g += self.l2_reg * w.detach() / self.n
            g += self.l2_reg * w.detach()
        return g
    
    @torch.no_grad()
    def get_dual_variables(self, losses):
        return get_smooth_weights(
            losses, self.sigmas, self.smooth_coef, self.smoothing
        )
    
    def compute_proximal_operator(self, q, losses, beta):
        n = len(losses)
        nu = self.smooth_coef
        new_smooth_coef = nu * (1 + beta)
        loss_vec = losses - nu * beta / n + nu * beta * q
        return get_smooth_weights(loss_vec, self.sigmas, new_smooth_coef, smoothing=self.smoothing)

    def get_indiv_loss(self, w, with_grad=False):
        if with_grad:
            return self.loss(w, self.X, self.y)
        else:
            with torch.no_grad():
                return self.loss(w, self.X, self.y)

    @torch.no_grad()
    def get_indiv_grad(self, w, X=None, y=None):
        if not (X is None):
            return self.grad_batch(w, X, y)
        else:
            return self.grad_batch(w, self.X, self.y)

    def get_model_cfg(self):
        return {
            "objective": self.risk_name,
            "l2_reg": self.l2_reg,
            "loss": self.loss_name,
            "n_class": self.n_class,
            "sm_coef": self.smooth_coef,
        }
    
class DivergenceBallObjective:
    def __init__(
        self,
        X,
        y,
        radius,
        loss="squared_error",
        l2_reg=None,
        n_class=None,
        risk_name=None,
        dataset=None,
        sm_coef=1.0,
        smoothing="l2",
        autodiff=False
    ):
        self.X = X
        self.y = y
        self.n, self.d = X.shape
        self.radius = radius
        self.loss = get_loss(loss, n_class=n_class)
        self.grad_batch = get_grad_batch(loss, n_class=n_class)
        self.loss_name = loss
        self.n_class = n_class
        self.l2_reg = l2_reg
        self.autodiff = autodiff

        # For logging.
        self.loss_name = loss
        self.risk_name = risk_name
        self.dataset = dataset

        self.smooth_coef = sm_coef
        self.smoothing = smoothing
        self.penalty = self.smooth_coef * self.radius
        self.penalty = 0.0

    @torch.no_grad()
    def get_batch_loss(self, w, include_reg=True):
        losses = self.loss(w, self.X, self.y)
        n = self.n
        q = self.get_dual_variables(losses)
        risk = torch.dot(q, losses) - 0.5 * self.smooth_coef * torch.sum((q - 1 / n) ** 2)
    
        if self.l2_reg and include_reg:
            risk += 0.5 * self.l2_reg * torch.norm(w) ** 2
        return risk

    @torch.no_grad()
    def get_batch_subgrad(self, w, idx=None, include_reg=True):
        if not (idx is None):
            X, y = self.X[idx], self.y[idx]
        else:
            X, y = self.X, self.y
        losses = self.loss(w, X, y)
        q = self.get_dual_variables(losses)
        g = torch.matmul(q, self.grad_batch(w, X, y))
        if self.l2_reg and include_reg:
            g += self.l2_reg * w.detach()
        return g

    def get_indiv_loss(self, w, with_grad=False):
        if with_grad:
            return self.loss(w, self.X, self.y)
        else:
            with torch.no_grad():
                return self.loss(w, self.X, self.y)

    @torch.no_grad()
    def get_dual_variables(self, losses):
        return torch.from_numpy(chi2_divergence_oracle(self.radius, self.smooth_coef, losses.numpy()))
    
    def compute_proximal_operator(self, q, losses, beta):
        nu = self.smooth_coef
        new_smooth_coef = nu * (1 + beta)
        loss_vec = losses + nu * beta * q
        return torch.from_numpy(chi2_divergence_oracle(self.radius, new_smooth_coef, loss_vec.numpy()))
    
    @torch.no_grad()
    def get_indiv_grad(self, w, X=None, y=None):
        # TODO: Check that these work
        if not (X is None):
            return self.grad_batch(w, X, y)
        else:
            return self.grad_batch(w, self.X, self.y)

    def get_model_cfg(self):
        return {
            "objective": self.risk_name,
            "l2_reg": self.l2_reg,
            "loss": self.loss_name,
            "n_class": self.n_class,
            "sm_coef": self.smooth_coef,
        }