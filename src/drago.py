import torch
import math
import numpy as np
import sys

sys.path.extend([".", ".."])
from src.dual import get_smooth_weights
from src.baselines import Optimizer


class Drago(Optimizer):


    @torch.no_grad()
    def __init__(
        self,
        objective,
        lr=0.01,
        seed_block=25,
        epoch_len=None,
        block_size=None,
        sm_coef=1.0,
        matched=False,
        cyclic=False,
        interval = 1
    ):
        super(Drago, self).__init__()
        self.objective = objective
        self.l2_reg = self.objective.l2_reg
        n, d = self.objective.n, self.objective.d
        if block_size is None:
            self.block_size = n // d
        elif block_size == "auto":
            self.block_size = min(n - 1, 16)
        else:
            self.block_size = block_size
        self.n_blocks = math.ceil(n / self.block_size)
        self.t = 0
        self.lr = lr
        self.b_const = 1 / (16 * lr * (1 + lr) * (n - 1) ** 2) # essentially 0.0
        self.rng_block = np.random.RandomState(seed_block)
        self.sm_coef = sm_coef
        self.matched = matched
        self.cyclic = cyclic
        self.interval = interval

        # primal variables
        if objective.n_class:
            self.weights = torch.zeros(objective.n_class * d, requires_grad=True, dtype=torch.float64)
            self.ws = torch.zeros(
                self.n_blocks, objective.n_class * d, requires_grad=False, dtype=torch.float64
            )
            self.gh1 = torch.zeros(n, objective.n_class * d, requires_grad=False, dtype=torch.float64)
            self.gh2 = torch.zeros(n, objective.n_class * d, requires_grad=False, dtype=torch.float64)
        else:
            self.weights = torch.zeros(d, requires_grad=True, dtype=torch.float64)
            self.ws = torch.zeros(
                self.n_blocks, d, requires_grad=False, dtype=torch.float64
            )
            self.gh1 = torch.zeros(n, d, requires_grad=False, dtype=torch.float64)
            self.gh2 = torch.zeros(n, d, requires_grad=False, dtype=torch.float64)

        # dual variables
        self.q =  torch.ones(n, requires_grad=False, dtype=torch.float64) / n

        # table of losses, gradients, and dual weights
        self.qh1 = torch.ones(n, requires_grad=False, dtype=torch.float64) / n
        self.qh2 = torch.ones(n, requires_grad=False, dtype=torch.float64) / n

        self.lh = self.objective.get_indiv_loss(self.weights)
        self.lh1 = self.lh.clone()
        self.gh1 = self.objective.get_indiv_grad(self.weights)
        self.gh2 = self.gh1.clone()

        # self.lh =  torch.zeros(n, requires_grad=False, dtype=torch.float64)
        # self.lh1 = torch.zeros(n, requires_grad=False, dtype=torch.float64)
        # for i in range(n):
        #     with torch.enable_grad():
        #         loss = self.objective.loss(
        #             self.weights, self.objective.X[i, :], self.objective.y[i]
        #         )
        #         g = torch.autograd.grad(outputs=loss, inputs=self.weights)[0]
        #     self.lh[i] = loss.item()
        #     self.lh1[i] = loss.item()
        #     self.gh1[i].copy_(g)
        #     self.gh2[i].copy_(g)
        self.g_agg = torch.matmul(self.gh1.T, self.qh1)
        self.w_agg = torch.sum(self.ws, dim=0)

        if epoch_len:
            self.epoch_len = epoch_len
        else:
            self.epoch_len = self.objective.n

        # used for block computations of gradients all at once
        # ft_compute_grad = grad(self._compute_loss)
        # self.ft_compute_sample_grad = vmap(ft_compute_grad, in_dims=(None, 0, 0))
        self.ft_compute_sample_grad = self.objective.get_indiv_grad
        self.n_oracle_evals = n

        self.prev_j_block = -1
        self.prev_k_block = -1

    def start_epoch(self):
        pass

    @torch.no_grad()
    def step(self):

        # sample block indices
        n = self.objective.n
        n_blocks, block_size = self.n_blocks, self.block_size
        k_block = self.t % self.n_blocks
        if self.cyclic:
            i_block = k_block
            j_block = k_block
        else:
            i_block = torch.tensor([self.rng_block.randint(0, n_blocks)]).item()
            if self.matched:
                # really, it needs to be matched to the last iteration
                j_block = i_block
            else:
                j_block = torch.tensor([self.rng_block.randint(0, n_blocks)]).item()

        # increment counter after selecting block, so the first block counter is 0
        self.t += 1

        # compute primal gradient estimate of coupled term
        delp = 0.0
        i_idx = torch.arange(i_block * block_size, min(n, (i_block + 1) * block_size))
        x, y = self.objective.X[i_idx, :],  self.objective.y[i_idx]
        # with torch.enable_grad():
        g1 = self.ft_compute_sample_grad(self.weights, x, y)
        delp += (torch.matmul(self.q[i_idx], g1) - torch.matmul(self.qh2[i_idx], self.gh2[i_idx])) / block_size
        vp = self.g_agg + n * delp / (1 + self.lr)
        if not (i_block in [self.prev_j_block, self.prev_k_block]):
            self.n_oracle_evals += len(i_idx)

        # perform primal update
        bt = (1 - (1 + self.lr) ** (1 - self.t)) / (self.lr * (1 + self.lr)) # essentially 1.0 / lr
        mu = self.l2_reg
        denom = mu * (1 + bt)
        numer = mu * (bt - (n_blocks - 1) * self.b_const) * self.weights + mu * self.b_const * self._get_sum(k_block - 1) - vp
        self.w_agg += numer / denom - self.ws[k_block] # replace this block in our running sum
        self.ws[k_block] = numer / denom
        self.weights.copy_(self.ws[k_block])


        # update loss table
        k_idx = torch.arange(k_block * block_size, min(n, (k_block + 1) * block_size))
        x, y = self.objective.X[k_idx, :],  self.objective.y[k_idx]
        losses_k = self._compute_loss(self.weights, x, y) # ideally we would not compute this twice
        # with torch.enable_grad():
        grads = self.ft_compute_sample_grad(self.weights, x, y)
        self.n_oracle_evals += len(k_idx)
            
        # perform dual update
        if self.t <= 30 or (self.t - 1) % self.interval == 0:
            j_idx = torch.arange(j_block * block_size, min(n, (j_block + 1) * block_size))
            if j_block == k_block:
                losses_j = losses_k
            else:
                x, y = self.objective.X[j_idx, :],  self.objective.y[j_idx]
                losses_j = self._compute_loss(self.weights, x, y)
            deld = (losses_j - self.lh1[j_idx]) / block_size
            vd = self.lh.clone()
            vd[j_idx] += n * deld / (1 + self.lr)
            vd[k_idx] += losses_k - self.lh[k_idx] # as if we updated self.lh first
            # self.q = self._compute_dual_proximal_step(self.q, vd, bt)
            self.q = self.objective.compute_proximal_operator(self.q, vd, bt)
        self.qh2[k_idx] = self.qh1[k_idx]
        self.qh1[k_idx] = self.q[k_idx]
        if j_block != k_block:
            self.n_oracle_evals += len(j_idx)

        # update loss and dual weight table
        self.gh2[k_idx] = self.gh1[k_idx]
        self.gh1[k_idx] = grads
        self.lh1[k_idx] = self.lh[k_idx]
        self.lh[k_idx]  = losses_k
        self.g_agg += torch.matmul(self.qh1[k_idx], self.gh1[k_idx]) - torch.matmul(self.qh2[k_idx], self.gh2[k_idx])

        # record gradient evaluations
        self.prev_j_block = j_block
        self.prev_k_block = k_block

    def _get_sum(self, block):
        return self.w_agg - self.ws[block]

    def _compute_loss(self, weights, x, y):
        return self.objective.loss(weights, x, y)

    # def _compute_dual_proximal_step(self, q, vd, bt):
    #     # this is a hack to use old pav code
    #     n = len(vd)
    #     spectrum = self.objective.sigmas
    #     nu = self.sm_coef
    #     smooth_coef = nu * (1 + bt)
    #     loss_vec = vd - nu * bt / n + nu * bt * q
    #     return get_smooth_weights(loss_vec, spectrum, smooth_coef, smoothing="l2")

    def end_epoch(self):
        pass

    def get_epoch_len(self):
        return self.epoch_len
    
    def get_oracle_evals(self):
        return self.n_oracle_evals