# drago
Code and experiments for "Drago: Primal-Dual Coupled Variance Reduction for Faster Distributionally Robust Optimization".

## Abstract
We consider the penalized distributionally robust optimization (DRO) problem with a closed, convex uncertainty set, a setting that encompasses learning using $f$-DRO and spectral/$L$-risk minimization. We present Drago, a stochastic primal-dual algorithm which combines cyclic and randomized components with a carefully regularized primal update to achieve dual variance reduction. Owing to its design, Drago enjoys a state-of-the-art linear convergence rate on strongly convex-strongly concave DRO problems witha fine-grained dependency on primal and dual condition numbers. The theoretical results are supported with numerical benchmarks on regression and classification tasks.

## Quickstart

You may use the implementation of Drago via the following steps.

### Dependencies

The required software environment can be build and activated with Anaconda/Miniconda using the following.
```
conda env create -f environment.yml
conda activate dro
```
The environment `dro` contains the necessary packages and Python version (3.10). We recommend a hardware environment has at least 32GB CPU RAM and a GPU with at least 12GB RAM. In addition, please install PyTorch following the [installation instructions](https://pytorch.org/get-started/locally/) for your particular CUDA distribution. For example, for CUDA 11.8, run:
```
pip3 install torch --index-url https://download.pytorch.org/whl/cu118
```
### Datasets

Five of the seven datasets `yacht`, `energy`, `concrete`, `kin8nm`, `power` are downloaded automatically when the are loaded (see `tutorial.ipynb`). The `emotion` dataset involved fine-tuning a pre-trained BERT model to generate embeddings. These are included directly in the package so that this step does not need to be repeated. Similarly, additional preprocessing is applied to `acsincome`. You may repeat these steps by running `download_acsincome.py`, or use the version that is already included in the `data` folder.

Finally, reproducing the `amazon` dataset requires fine-tuning a BERT model to produce frozen feature representations. This can be done by running through the entirety of the `download_amazon.ipynb` notebook. A preprocessed version already exists in the `data/amazon` folder.

### Tutorial

After completing all of the above steps, see `tutorial.ipynb` for a walkthrough of the code structure and how to reproduce experimental results.
