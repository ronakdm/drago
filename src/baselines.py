import torch
import numpy as np
import sys

sys.path.extend([".", ".."])
from src.dual import get_smooth_weights_sorted

class Optimizer:
    def __init__(self):
        pass

    def start_epoch(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError

    def end_epoch(self):
        raise NotImplementedError

    def get_epoch_len(self):
        raise NotImplementedError
    
    def get_oracle_evals(self):
        raise NotImplementedError
    
class StochasticSubgradientMethod(Optimizer):
    def __init__(self, objective, lr=0.01, batch_size=64, seed=25, epoch_len=None, momentum=None):
        super(StochasticSubgradientMethod, self).__init__()
        self.objective = objective
        self.lr = lr
        self.batch_size = batch_size
        self.n = self.objective.n

        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * self.objective.d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            self.weights = torch.zeros(
                self.objective.d, requires_grad=True, dtype=torch.float64
            )
        self.order = None
        self.iter = 0
        self.momentum_param = momentum
        if momentum:
            self.momentum = torch.zeros(self.objective.d, dtype=torch.float64)
        torch.manual_seed(seed)
        np.random.seed(seed)

        if epoch_len:
            self.epoch_len = min(epoch_len, self.objective.n // self.batch_size)
        else:
            self.epoch_len = self.objective.n // self.batch_size
        
        self.n_oracle_evals = 0
        self.order = torch.from_numpy(np.random.permutation(self.objective.n))
        self.batch_idx = 0

    def start_epoch(self):
        # self.order = torch.from_numpy(np.random.permutation(self.objective.n))
        self.iter = 0

    @torch.no_grad()
    def step(self):
        n = self.n
        if self.batch_idx * self.batch_size >= n:
            self.order = torch.from_numpy(np.random.permutation(self.objective.n))
            self.batch_idx = 0

        idx = self.order[
            self.batch_idx
            * self.batch_size : min(self.n, (self.batch_idx + 1) * self.batch_size)
        ]
        self.batch_idx += 1
        
        g = self.objective.get_batch_subgrad(self.weights, idx=idx)
        self.n_oracle_evals += len(idx)

        if self.momentum_param:
            self.momentum *= self.momentum_param
            self.momentum += g
            self.weights.copy_(self.weights - self.lr * self.momentum)
        else:
            self.weights.copy_(self.weights - self.lr * g)
        self.iter += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len
    
    def get_oracle_evals(self):
        return self.n_oracle_evals
    

class LSVRG(Optimizer):
    def __init__(
        self,
        objective,
        lr=0.01,
        uniform=True,
        nb_passes=1,
        smooth_coef=1.0,
        smoothing="l2",
        seed=25,
        length_epoch=None,
    ):
        super(LSVRG, self).__init__()
        n, d = objective.n, objective.d
        self.objective = objective
        self.lr = lr
        if objective.n_class:
            self.weights = torch.zeros(
                objective.n_class * d,
                requires_grad=True,
                dtype=torch.float64,
            )
        else:
            self.weights = torch.zeros(d, requires_grad=True, dtype=torch.float64)
        self.spectrum = self.objective.sigmas
        self.rng = np.random.RandomState(seed)
        self.uniform = uniform
        # self.smooth_coef = n * smooth_coef if smoothing == "l2" else smooth_coef
        self.smooth_coef = smooth_coef
        self.smoothing = smoothing
        if length_epoch:
            self.length_epoch = length_epoch
        else:
            self.length_epoch = int(nb_passes * n)
        self.nb_checkpoints = 0
        self.n_oracle_evals = 0
        self.step_no = 0

    @torch.no_grad()
    def start_epoch(self):
        pass

    @torch.no_grad()
    def step(self):
        n = self.objective.n

        # start epoch
        if self.step_no % n == 0:
            losses = self.objective.get_indiv_loss(self.weights, with_grad=False)
            sorted_losses, self.argsort = torch.sort(losses, stable=True)
            with torch.no_grad():
                self.sigmas = get_smooth_weights_sorted(
                    sorted_losses, self.spectrum, self.smooth_coef, self.smoothing
                )

            self.subgrad_checkpt = self.objective.get_batch_subgrad(self.weights, include_reg=False)
            self.weights_checkpt = torch.clone(self.weights)
            self.nb_checkpoints += 1
            self.n_oracle_evals += self.objective.n

        if self.uniform:
            i = torch.tensor([self.rng.randint(0, n)])
        else:
            i = torch.tensor([np.random.choice(n, p=self.sigmas)])
        x = self.objective.X[self.argsort[i]]
        y = self.objective.y[self.argsort[i]]

        # Compute gradient at current iterate.
        g = self.objective.get_indiv_grad(self.weights, x, y)
        g_checkpt = self.objective.get_indiv_grad(self.weights_checkpt, x, y)

        # we do not count the second one, as if we used a high memory variant
        self.n_oracle_evals += 1 

        if self.uniform:
            direction = n * self.sigmas[i] * (g - g_checkpt) + self.subgrad_checkpt
        else:
            direction = g - g_checkpt + self.subgrad_checkpt
        if self.objective.l2_reg:
            direction += self.objective.l2_reg * self.weights
        self.weights.copy_(self.weights - self.lr * direction.reshape(-1))
        self.step_no += 1

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.length_epoch
    
    def get_oracle_evals(self):
        return self.n_oracle_evals