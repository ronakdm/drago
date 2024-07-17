# drago
Code and experiments for "Drago: Primal-Dual Coupled Variance Reduction for Faster Distributionally Robust Optimization".

## Abstract
We consider the penalized distributionally robust optimization (DRO) problem with a closed, convex uncertainty set, a setting that encompasses learning using $f$-DRO and spectral/$L$-risk minimization. We present Drago, a stochastic primal-dual algorithm which combines cyclic and randomized components with a carefully regularized primal update to achieve dual variance reduction. Owing to its design, Drago enjoys a state-of-the-art linear convergence rate on strongly convex-strongly concave DRO problems witha fine-grained dependency on primal and dual condition numbers. The theoretical results are supported with numerical benchmarks on regression and classification tasks.

## Background

We specifically consider optimization problems of the form

$$
    \min_{w \in \mathcal{W}} \max_{q \in \mathcal{Q}} \mathcal{L}(w, q) := \sum_{i=1}^n q_i \ell_i(w) - \nu D(Q \Vert \mathbb{1}/n) + \frac{\mu}{2} \Vert w \Vert_2^2
$$

where $(w, q)$ denotes a primal-dual pair, $\ell_i$ denotes the training loss on example $i$, and $D$ denotes statistical divergence such as KL or $\chi^2$.

## Dependencies and Quickstart

The required software environment can be build and activated with Anaconda/Miniconda using the following.
```
conda env create -f environment.yml
conda activate dro
```
The environment `dro` contains the necessary packages and Python version (3.10). We recommend a hardware environment has at least 32GB CPU RAM and a GPU with at least 12GB RAM. In addition, please install PyTorch following the [installation instructions](https://pytorch.org/get-started/locally/) for your particular CUDA distribution. For example, for CUDA 11.8, run:
```
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```
GPUs are not necessary to run the various algorithms. After setting up the environment, see `example.ipynb` for a "quickstart" walkthrough of the code structure and how to run the algorithms. Example data is downloaded in the process of loading (see below), but this can be replaced with your own data.

## Datasets

There are seven total datasets used in the numerical benchmarks.

| Dataset Tag | $n$ | $d$ | Source |
| ----------- | ----------- | ----------- | ----------- |
| `yacht`     | 244  | 6 | UCI |
| `energy`    | 614  | 8 | UCI |
| `concrete`  | 824  | 8 | UCI |
| `kin8nm`    | 6,553  | 8 | OpenML |
| `power`     | 7,654  | 4 | UCI |
| `acsincome` | 4,000  | 202 | Fairlearn |
| `emotion`   | 8,000  | 270 | Hugging Face |

Five of the seven datasets `yacht`, `energy`, `concrete`, `kin8nm`, `power` are downloaded automatically when the are loaded (see `example.ipynb`). The `emotion` dataset involved fine-tuning a pre-trained BERT model to generate embeddings. These are included directly in the package so that this step does not need to be repeated, but this can be reproduced by following `notebooks/create_emotion.ipynb`. In this case, ensure that Hugging Face libraries are installed after installing PyTorch, via:
```
pip install transformers
pip install datasets
```
In addition, the version of `"emotion"` used in the paper has since been removed from Hugging Face - we use the equivalent copy `"dair-ai/emotion"` in the example to demonstrate the steps. Similarly, additional preprocessing is applied to `acsincome`. You may repeat these steps by running `download_acsincome.py`, or use the version that is already included in the `data` folder.

