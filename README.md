# Poisson reduced-rank regression for modeling neural activity
## Overview
`poisson_rrr.py` contains the implementation of the Poisson reduced-rank regression (p-RRR) used in the manuscript [Global and local origins of trial-to-trial spike count variability in visual cortex](https://www.biorxiv.org/content/10.1101/2025.08.08.669442v1). `linear_rrr.py` contains the implementation of the [linear reduced-rank regression (RRR)](https://andrewcharlesjones.github.io/journal/reduced-rank-regression.html) which has been used to analyze [continuous neural data](https://www.science.org/doi/10.1126/science.aav7893). `poissonrrr_demo.ipynb` provides a quick demo (< 5mins to run) comparing the two models in a toy example with simulated spikes matrix. 

Briefly, p-RRR extends RRR to model discrete counting data, such as binned spike trains. Informally, for $X$, an $N \times P$ matrix of predictors, and $Y$, an $N \times Q$ matrix of targets, RRR models the relationship $Y \sim \text{Normal}(XW+b, \sigma^2), \text{with } \text{rank}(W)\leq r$. p-RRR models $Y \sim \text{Poisson}(f(XW+b)), \text{with } \text{rank}(W)\leq r$.


In our study, we used p-RRR to model the target neural population activities as a function of the other recorded variables (stimuli, facial behavior, activity of another neural population, etc), while enabling a shared structure on the dependence through its rank constraint. When its rank is not constrained, p-RRR is equivalent to fitting separate [Poisson GLMs](https://www.sciencedirect.com/science/article/pii/S0079612306650310) to each neuron in the target population individually. Similar to Poisson GLM, p-RRR also needs to be fitted using iterative numerical methods. We implemented p-RRR in [PyTorch](https://pytorch.org/) as a simple feed-forward neural network; its rank constraint is enforeced with a bottleneck hidden layer. 

## Installation
### 1. Clone this repo to your machine
```
git clone https://github.com/ziyulu-uw/poissonrrr.git
cd poissonrrr
```
### 2. Install
p-RRR can be fitted on both CPU and GPU through PyTorch. GPU is not necessary to run the toy example here.

#### 2.1. CPU-only version (~5mins to install)
- Using Conda (tested):
```
conda env create -f environment.yml
conda activate poissonrrr
jupyter lab  # open jupyter lab to run poissonrrr_demo.ipynb
```
- Using pip + venv
```
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install torch==2.0.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
jupyter lab  # open jupyter lab to run poissonrrr_demo.ipynb
```
#### 2.2. If running on GPU is desired, need to install GPU version of PyTorch separately outside of requirements.txt. (~15mins to install)

Example commands for Conda:
```
conda create -n poissonrrr python=3.10 -c conda-forge
conda activate poissonrrr

# Install PyTorch 2.0.0 GPU build:
# Visit https://pytorch.org/get-started/previous-versions/ and copy the command there
# for PyTorch 2.0.0, your OS, and CUDA version. Example for CUDA 11.8:
conda install pytorch==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

# Install remaining dependencies
pip install -r requirements.txt

jupyter lab
```
