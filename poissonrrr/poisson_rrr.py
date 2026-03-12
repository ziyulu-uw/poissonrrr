"""
Poisson reduced-rank regression utilities built on top of PyTorch.

This module defines:
- `GLM`: a generalized linear model that can operate in full-rank mode
    (`rank=0`) or reduced-rank mode (`rank>0`) via a two-layer factorization.
- `PoissonRRR`: a training/inference/evaluation wrapper that handles
    predictor normalization, LBFGS optimization, optional predictor-specific
    L2 regularization, and common decoding metrics.

Supported activations are `softplus`, `exp`, and `relu`. Supported losses are
`poisson` and `mse`. When using `act='exp'` with Poisson loss, the model
optimizes in log-rate space for numerical stability.

Activation names are validated at initialization. Unsupported activations
raise `ValueError` rather than silently falling back to an inconsistent
forward-pass behavior.

Predictor inputs are expected as `Xs`, a list of 2D NumPy arrays, where each
array has the same number of rows (samples/observations).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import sklearn.metrics


VALID_ACTIVATIONS = {'softplus', 'exp', 'relu'}

class GLM(nn.Module):
    """
    Generalized linear model with optional reduced-rank weight factorization.

    In full-rank mode (`rank=0`), the model uses a single linear mapping from
    `n_input` to `n_output`. In reduced-rank mode (`rank>0`), it parameterizes
    the effective weight matrix as the product of two smaller matrices:
    `W_eff = W2 @ W1`, where `W1` maps input to a latent rank-dimensional
    subspace and `W2` maps that subspace to outputs.

    The forward pass applies one of the supported nonlinearities (`exp`,
    `relu`, `softplus`) to produce output rates/values.
    """

    def __init__(self, n_input, n_output, act='softplus', rank=0, loss='poisson', seed=-1, W0=None):
        """
        Initialize a GLM with optional reduced-rank structure.

        Parameters
        ----------
        n_input : int
            Number of input features.
        n_output : int
            Number of output units.
        act : str, default='softplus'
            Output nonlinearity. Supported values are `'softplus'`, `'exp'`,
            and `'relu'`.
        rank : int, default=0
            Rank constraint for the effective weight matrix. If `rank=0`, use a
            single full-rank linear layer. If `rank>0`, use a two-layer
            factorization with latent dimension `rank`.
        loss : str, default='poisson'
            Training loss identifier used to determine output interpretation in
            some branches. Supported values are `'poisson'` and `'mse'`.
        seed : int, default=-1
            Random seed for weight initialization. If negative, no seed is set.
        W0 : np.ndarray or None, default=None
            Optional initial full weight matrix with shape
            `(n_output, n_input)`. In full-rank mode, this directly initializes
            the linear layer weight. In reduced-rank mode, this matrix is
            decomposed with SVD and the top-`rank` right singular vectors are
            used to initialize `linear1.weight`, while the top-`rank` left
            singular vectors are used to initialize `linear2.weight`. The
            singular values are discarded, so this initialization preserves the
            leading subspace of `W0` but does not exactly reconstruct its scale.

        Notes
        -----
        In reduced-rank mode, `rank` must satisfy
        `rank <= min(n_input, n_output)`.

        Raises
        ------
        ValueError
            If `act` is not one of `'softplus'`, `'exp'`, or `'relu'`, or if
            `rank > min(n_input, n_output)` in reduced-rank mode.
        """
        super(GLM, self).__init__()
        if seed >= 0:
            torch.manual_seed(seed)
        if act not in VALID_ACTIVATIONS:
            raise ValueError(
                "Unsupported activation {!r}. Expected one of {}.".format(
                    act, sorted(VALID_ACTIVATIONS)
                )
            )
        self.act = act
        self.rank = rank
        self.loss = loss
        if self.rank == 0:  # full rank glm
            self.linear = nn.Linear(n_input, n_output)
            if W0 is not None:
                self.linear.weight.data = torch.from_numpy(W0)
                self.linear.bias.data.zero_()  # Keep initialization consistent with weight-only W0.
        else:  # reduced-rank glm
            if rank > min(n_input, n_output):
                raise ValueError(
                    'rank {} is invalid; rank must be less than or equal to {}'.format(
                        rank, min(n_input, n_output)
                    )
                )
            self.linear1 = nn.Linear(n_input, rank)
            self.linear2 = nn.Linear(rank, n_output)
            if W0 is not None:
                # Initialize factors from leading singular vectors of W0.
                u, s, vh = np.linalg.svd(W0, full_matrices=False)
                self.linear1.weight.data = torch.from_numpy(vh[:rank]).float()
                self.linear2.weight.data = torch.from_numpy(u[:,:rank]).float()
                self.linear1.bias.data.zero_()  # Keep initialization consistent with weight-only W0.
                self.linear2.bias.data.zero_()  # Keep initialization consistent with weight-only W0.

    def forward(self, x):
        """
        Run a forward pass through the GLM.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(n_samples, n_input)`.

        Returns
        -------
        torch.Tensor
            Output tensor of shape `(n_samples, n_output)`. For
            `act='exp'` with Poisson loss, this returns log-rates (not
            exponentiated rates) so `PoissonNLLLoss(log_input=True)` can be
            used stably.
        """
        if self.rank == 0:
            if self.act == 'exp':
                if self.loss == 'poisson':
                    x = self.linear(x)  # Return log-rates for stable PoissonNLLLoss(log_input=True).
                else:
                    x = torch.exp(self.linear(x))
            elif self.act == 'relu':
                x = F.relu(self.linear(x))
            elif self.act == 'softplus':
                x = F.softplus(self.linear(x))
        else:
            x = self.linear1(x)
            if self.act == 'exp':
                if self.loss == 'poisson':
                    x = self.linear2(x)  # Return log-rates for stable PoissonNLLLoss(log_input=True).
                else:
                    x = torch.exp(self.linear2(x))
            elif self.act == 'relu':
                x = F.relu(self.linear2(x))
            elif self.act == 'softplus':
                x = F.softplus(self.linear2(x))
        return x

class PoissonRRR:
    """
    Wrapper for training, inference, and evaluation of reduced-rank GLMs.

    This class manages:
    - predictor preprocessing via per-block mean/std normalization,
    - model fitting with LBFGS and optional block-specific L2 penalties,
    - prediction (including optional checkpoint loading), and
    - evaluation with Poisson and regression-style metrics.

    Normalization statistics (`meanXs`, `stdXs`) are fit on first use of
    `normalize` and then reused for subsequent calls (for example, inference on
    held-out data).
    """

    def __init__(self, n_input, n_output, act='softplus', rank=0, loss='poisson', zeromeanXs=None, normstdXs=None, regList=None, seed=-1, W0=None, verbose=False):
        """
        Initialize a PoissonRRR model wrapper.

        Parameters
        ----------
        n_input : int
            Total number of input features after concatenating all predictor
            blocks in `Xs`.
        n_output : int
            Number of output units (for example, neurons).
        act : str, default='softplus'
            Output nonlinearity used by the underlying GLM. Supported values are
            `'softplus'`, `'exp'`, and `'relu'`.
        rank : int, default=0
            Rank constraint for the GLM weight matrix. `0` means unconstrained
            full-rank.
        loss : str, default='poisson'
            Training loss identifier. Supported values are `'poisson'` and
            `'mse'`.
        zeromeanXs : list or None
            Per-predictor-block flag list. If `zeromeanXs[i]` is truthy, the
            i-th predictor block is mean-centered during normalization. If
            None, defaults to an empty list.
        normstdXs : list or None
            Per-predictor-block flag list. If `normstdXs[i]` is truthy, the
            i-th predictor block is scaled by its feature-wise standard
            deviation during normalization. If None, defaults to an empty list.
        regList : list or None
            Optional list of L2 regularization weights per predictor block.
            `regList[i]` applies to the i-th block in the concatenated input.
            Defaults to an empty list. Passing `None` or `[]` disables L2
            regularization.
        seed : int, default=-1
            Random seed for model initialization/training behavior. If negative,
            no seed is set.
        W0 : np.ndarray or None, default=None
            Optional initial full weight matrix passed to `GLM` initialization.
        verbose : bool, default=False
            Whether to print auxiliary runtime information such as selected
            device information during training and prediction.
        """
        self.n_input = n_input
        self.n_output = n_output
        self.act = act
        self.rank = rank
        self.loss = loss.lower()
        zeromeanXs = [] if zeromeanXs is None else zeromeanXs
        normstdXs = [] if normstdXs is None else normstdXs
        regList = [] if regList is None else regList
        self.regList = regList
        self.zeromeanXs= zeromeanXs
        self.normstdXs = normstdXs
        self.seed = seed
        self.verbose = verbose
        self.meanXs = None
        self.stdXs = None
        # define model
        self.glm = GLM(n_input=n_input, n_output=n_output, act=act, rank=rank, loss=loss, seed=seed, W0=W0)
        # define loss
        if self.loss == 'poisson':
            if self.act == 'exp':
                # `forward` returns log-rates in this branch, so use log_input=True.
                self.loss_func = nn.PoissonNLLLoss(log_input=True, full=True, reduction='mean')
            else:
                self.loss_func = nn.PoissonNLLLoss(log_input=False, full=True, reduction='mean')
        elif self.loss == 'mse':
            self.loss_func = nn.MSELoss()

    def normalize(self, Xs):
        """
        Concatenate and normalize predictor blocks using stored train statistics.

        On first call, this method computes and stores per-feature means and
        standard deviations according to `zeromeanXs` and `normstdXs`.
        Subsequent calls reuse the stored values so evaluation/inference use the
        same normalization as training.

        Parameters
        ----------
        Xs : list of np.ndarray
            Predictor blocks. Each element must be a 2D array with the same
            number of rows (samples).

        Returns
        -------
        np.ndarray
            Concatenated and normalized predictor matrix of shape
            `(n_samples, n_input)`.

        Raises
        ------
        ValueError
            If the normalized predictor matrix contains non-finite values.
        """
        if self.meanXs is None:  # compute mean & std for the first time (on the training data)
            meanXs = np.array([])
            stdXs = np.array([])

            for i, X in enumerate(Xs):  # prepare each predictor separately
                if self.zeromeanXs[i]:
                    meanX = np.mean(X, axis=0)
                else:
                    meanX = np.zeros(X.shape[1])

                if self.normstdXs[i]:
                    stdX = np.std(X, axis=0)
                else:
                    stdX = np.ones(X.shape[1])

                meanXs = np.append(meanXs, meanX)
                stdXs = np.append(stdXs, stdX)

            # Avoid divide-by-zero when a feature is constant in the training set.
            zerostdix = np.argwhere(stdXs < np.spacing(1))
            if len(zerostdix) > 0:
                stdXs[zerostdix.ravel()] = 1  # Constant features get unit scaling so normalization does not divide by zero.

            self.meanXs = meanXs
            self.stdXs = stdXs

        Xall = np.hstack(Xs)
        Xall = Xall - np.tile(self.meanXs, (Xall.shape[0], 1))
        Xall = np.divide(Xall, np.tile(self.stdXs, (Xall.shape[0], 1)))

        # Make sure all normalized data are finite before proceeding.
        if np.any(~np.isfinite(Xall)):
            raise ValueError('invalid value found in data, cannot proceed')

        return Xall

    def train(self,
              Xs_train,
              Y_train,
              lr=1,
              maxepoch=100,
              max_iter=10,
              history_size=10,
              line_search=1,
              grad_tol=1e-4,
              loss_tol=1e-6,
              patience=10,
              shuffle=True,
              track=10,
              progfile=None,
              modelfile=None,
              usegpu=False):
        """
        Fit the model parameters using LBFGS optimization.

        Parameters
        ----------
        Xs_train : list of np.ndarray
            Training predictor blocks. Each array must have the same number of
            rows (samples).
        Y_train : np.ndarray
            Training targets with the same number of rows as each predictor
            block in `Xs_train`.
        lr : float, default=1
            LBFGS learning rate.
        maxepoch : int, default=100
            Maximum number of outer training epochs.
        max_iter : int, default=10
            Maximum LBFGS iterations per optimizer step.
        history_size : int, default=10
            LBFGS history size.
        line_search : int or bool, default=1
            If truthy, enable strong-Wolfe line search.
        grad_tol : float, default=1e-4
            Stop when the maximum parameter gradient magnitude falls below this
            threshold.
        loss_tol : float, default=1e-6
            Improvement threshold used to count low-improvement epochs.
        patience : int, default=10
            Stop after this many total epochs (cumulative count within a run)
            where loss improvement is below `loss_tol`.
        shuffle : bool, default=True
            Whether to shuffle training samples between epochs.
        track : int, default=10
            Progress logging period in epochs.
        progfile : file-like or None, default=None
            Optional writable stream for progress logs. If None, logs are printed.
        modelfile : str or None, default=None
            Optional path to save model state and normalization statistics.
        usegpu : bool, default=False
            Whether to attempt GPU acceleration (MPS on macOS or CUDA).

        Returns
        -------
        tuple[list, list]
            `(train_loss_hist, grad_hist)` containing training loss history and
            per-epoch gradient summaries.
        """
        # set random seed if any
        if self.seed >= 0:
            rng = np.random.default_rng(seed=self.seed)
            torch.manual_seed(self.seed)
        else:
            rng = np.random.default_rng()

        # use gpu if possible
        device = "cpu"
        if usegpu and torch.backends.mps.is_available():
            device = torch.device("mps")  # for mac
        elif usegpu and torch.cuda.is_available():
            device = 'cuda:0'
        if self.verbose:
            print('train:', device)

        X_train = torch.from_numpy(self.normalize(Xs_train)).float()
        Y_train = torch.from_numpy(Y_train).float()
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)
        self.glm.to(device)

        # define optimization method
        line_search_fn = 'strong_wolfe' if line_search else None
        optimizer = optim.LBFGS(self.glm.parameters(), lr=lr, history_size=history_size, max_iter=max_iter,
                                line_search_fn=line_search_fn)

        # define regularization
        # since regularization weights may differ between predictors, here we define the "boundaries" for each regularization weight
        if self.regList:
            cumdim = np.cumsum([Xs_train[i].shape[1] for i in range(len(Xs_train))])
            cumdim = np.insert(cumdim, 0, 0)

        train_loss_hist = []
        grad_hist = []

        stop = False
        e = 0
        cnt = 0

        # compute initial loss before training
        with torch.no_grad():
            Y_train_pred = self.glm(X_train)
            train_loss_hist.append(self.loss_func(Y_train_pred, Y_train).item())

        if progfile is not None:
            toprint = 'epoch {}: training loss {:.6f} \n'.format(e, train_loss_hist[-1])
            progfile.write(toprint)
        else:
            print('epoch {}: training loss \n'.format(e), train_loss_hist[-1])

        while not stop:
            e += 1
            # Assumes up to four parameter tensors (reduced-rank case).
            # In full-rank mode, unused slots remain zero and do not affect max.
            grad_step = torch.zeros(4).to(device)

            if shuffle:
                shuffled_ixs = rng.permutation(X_train.shape[0])
                X_train = X_train[shuffled_ixs]
                Y_train = Y_train[shuffled_ixs]

            # LBFGS reevaluates this closure multiple times per step.
            def closure():
                optimizer.zero_grad()
                Y_train_hat = self.glm(X_train)
                loss = self.loss_func(Y_train_hat, Y_train)
                if self.regList:  # apply separate L2 regularization each predictor
                    for k in range(len(Xs_train)):
                        if self.rank > 0:
                            loss += self.regList[k] * \
                                    torch.sum((self.glm.linear2.weight @ self.glm.linear1.weight[:,
                                                                         cumdim[k]:cumdim[k + 1]]) ** 2)
                        else:
                            loss += self.regList[k] * torch.sum(
                                (self.glm.linear.weight[:, cumdim[k]:cumdim[k + 1]]) ** 2)

                loss.backward()
                return loss

            optimizer.step(closure)
            # Re-run the closure once so param.grad reflects the current iterate
            # for gradient tracking and stopping checks after LBFGS finishes.
            loss = closure()

            for l, param in enumerate(self.glm.parameters()):
                grad_step[l] += torch.max(abs(param.grad))

            # record keeping and check if any early stopping criterion is met
            train_loss_hist.append(loss.item())
            grad_hist.append(grad_step.cpu().numpy())

            if torch.max(grad_step) < grad_tol:
                stop = True
                if progfile is not None:
                    toprint = 'gradient less than {} after {} epochs \n '.format(grad_tol, e)
                    progfile.write(toprint)
                else:
                    print('gradient less than {} after {} epochs \n '.format(grad_tol, e))

            if e > 1:
                if train_loss_hist[-2] - train_loss_hist[-1] < loss_tol:
                    cnt += 1

            if cnt == patience:
                stop = True
                if progfile is not None:
                    progfile.write('{} loss changes less than {} after {} epochs \n '.format(patience, loss_tol, e))
                else:
                    print('{} loss changes less than {} after {} epochs \n '.format(patience, loss_tol, e))

            if e % track == 0:
                if progfile is not None:
                    progfile.write('{} epochs: train loss {:.6f} \n'.format(e, train_loss_hist[-1]))
                    progfile.flush()
                else:
                    print('epoch {}: training loss'.format(e), train_loss_hist[-1])
                    print('          gradient {}'.format(grad_hist[-1]))

            if e == maxepoch:
                stop = True
                if progfile is not None:
                    progfile.write('reached maximum epoch {} \n'.format(maxepoch))
                else:
                    print('reached maximum epoch', maxepoch)

        if progfile is not None:
            progfile.write('final train loss {:.6f} grad {} \n'.format(train_loss_hist[-1], grad_hist[-1]))
        else:
            print('final train loss {:.6f} grad {} \n'.format(train_loss_hist[-1], grad_hist[-1]))

        if modelfile:
            torch.save({'model_state_dict': self.glm.state_dict(), 'meanXs': self.meanXs, 'stdXs': self.stdXs}, modelfile)

        return train_loss_hist, grad_hist

    def predict(self, Xs, usegpu=False, modelfile=None):
        """
        Generate predictions from the current or a loaded model state.

        Parameters
        ----------
        Xs : list of np.ndarray
            Predictor blocks with the same row-count convention used in training.
        usegpu : bool, default=False
            Whether to attempt GPU inference (MPS on macOS or CUDA).
        modelfile : str or None, default=None
            Optional checkpoint path. If provided:
            - `.pth`: loads `model_state_dict`, `meanXs`, and `stdXs`.
            - `.npy`: loads a state-dict-like mapping into model parameters.

        Returns
        -------
        np.ndarray
            Predicted output matrix with shape `(n_samples, n_output)`. If
            `loss='poisson'` and `act='exp'`, the returned values are converted
            from log-rates to rates via `exp`.
        """
        if modelfile is not None:
            if modelfile.endswith('.pth'):
                model = torch.load(modelfile)
                self.meanXs = model['meanXs']
                self.stdXs = model['stdXs']
                self.glm.load_state_dict(model['model_state_dict'])
            elif modelfile.endswith('.npy'):
                model = np.load(modelfile)
                own_state = self.glm.state_dict()
                D = model.item()
                for name, param in D.items():
                    if name in own_state:
                        try:
                            own_state[name].copy_(param)
                        except Exception:
                            raise RuntimeError(
                                'While copying the parameter named {}, whose dimensions in the model are {} and whose ' \
                                'dimensions in the checkpoint are {}.'.format(name, own_state[name].size(), param.size()))
                    else:
                        raise KeyError('unexpected key "{}" in state_dict'.format(name))

        device = "cpu"
        if usegpu and torch.backends.mps.is_available():
            device = torch.device("mps")  # for mac
        elif usegpu and torch.cuda.is_available():
            device = 'cuda:0'
        if self.verbose:
            print(f'device: {device}')
        self.glm.to(device)

        with torch.no_grad():
            X_ts = torch.from_numpy(self.normalize(Xs)).float().to(device)
            Y_pred = self.glm(X_ts).cpu().numpy()

        if (self.loss == 'poisson') and (self.act == 'exp'):
            Y_pred = np.exp(Y_pred)

        return Y_pred

    def eval(self, Xs, Y, Y_pred=None, modelfile=None, usegpu=False, metrics=None, return_raw=False):
        """
        Evaluate predictions with Poisson and regression-style metrics.

        Parameters
        ----------
        Xs : list of np.ndarray
            Predictor blocks used for generating predictions when `Y_pred` is
            not provided.
        Y : np.ndarray
            Ground-truth output matrix with shape `(n_samples, n_output)`.
        Y_pred : np.ndarray or None, default=None
            Optional precomputed predictions. If None, `self.predict(...)` is
            called.
        modelfile : str or None, default=None
            Optional checkpoint path used when `Y_pred` is not provided.
        usegpu : bool, default=False
            Whether to use GPU when calling `predict`.
        metrics : list of str or None, default=None
            Metrics to compute. Supported values include:
            `'PNLL'`, `'MSE'`, `'d2'`, `'cc'`, and `'r2'`. If None,
            defaults to `['PNLL']`.
        return_raw : bool, default=False
            If True, include per-output raw metric arrays (for metrics that
            support them) in addition to aggregate scores.

        Returns
        -------
        tuple[dict, np.ndarray]
            `metrics_scores` and `Y_pred`. Depending on requested metrics,
            dictionary keys may include:
            `PNLL`, `MSE`, `d2uniform`, `d2nullweight`, `d2varweight`,
            `d2raw`, `cc`, `ccraw`, `r2weight`, `r2uniform`, and `r2raw`.
        """
        if Y_pred is None:
            Y_pred = self.predict(Xs, usegpu=usegpu, modelfile=modelfile)

        metrics = ['PNLL'] if metrics is None else metrics

        metrics_scores = {}

        # filter out neurons whose spike train std is 0. These neurons typically have a spike train of all zeros.
        # so it's neither interesting nor stable to compute scores like d2 and cc for them.
        Y_std = np.std(Y, axis=0)
        zerostdix = np.argwhere(Y_std == 0)
        validix = np.arange(Y.shape[1])
        mask = np.ones(Y.shape[1], dtype=bool)
        if len(zerostdix) > 0:
            mask[zerostdix.ravel()] = False
        validix = validix[mask]

        for m in metrics:
            if m.upper() == 'PNLL':
                loss_func = nn.PoissonNLLLoss(log_input=False, full=True, reduction='mean')
                test_loss = loss_func(torch.from_numpy(Y_pred).float(), torch.from_numpy(Y).float()).item()
                metrics_scores['PNLL'] = test_loss
            elif m.upper() == 'MSE':
                loss_func = nn.MSELoss()
                test_loss = loss_func(torch.from_numpy(Y_pred).float(), torch.from_numpy(Y).float()).item()
                metrics_scores['MSE'] = test_loss
            elif m == 'd2':
                eps = np.spacing(1)  # Keep predictions strictly positive for Poisson-based metrics.
                Y_pred_nonneg = np.clip(Y_pred, a_min=0, a_max=None) + eps
                d2_all = np.array([sklearn.metrics.d2_tweedie_score(Y[:, i], Y_pred_nonneg[:, i], power=1) for i in validix])
                d2uniform = np.average(d2_all)
                Y_null_pred = np.tile(np.mean(Y, axis=0), (Y.shape[0], 1))
                null_poisson_deviance = np.array(
                    [sklearn.metrics.mean_poisson_deviance(Y[:, i], Y_null_pred[:, i] + eps) for i in validix])
                d2nullweight = np.average(d2_all, weights=null_poisson_deviance)
                d2varweight = np.average(d2_all, weights=np.var(Y[:, validix], axis=0, ddof=1))
                metrics_scores['d2uniform'] = d2uniform
                metrics_scores['d2nullweight'] = d2nullweight
                metrics_scores['d2varweight'] = d2varweight
                if return_raw:
                    d2_all_raw = np.full([Y.shape[1]], np.nan)
                    d2_all_raw[validix] = d2_all
                    metrics_scores['d2raw'] = d2_all_raw
            elif m == 'cc':
                # filter out neurons whose predicted spike train has std=0. cc will throw error on them.
                Ypred_std = np.std(Y_pred, axis=0)
                zerostdix = np.argwhere(Ypred_std == 0)
                validix1 = np.arange(Y_pred.shape[1])
                mask = np.ones(Y_pred.shape[1], dtype=bool)
                if len(zerostdix) > 0:
                    mask[zerostdix.ravel()] = False
                validix1 = validix1[mask]
                validix2 = np.intersect1d(validix, validix1)

                cc_all = np.array([np.corrcoef(Y[:, i], Y_pred[:, i])[0, 1] for i in validix2])
                cc = np.average(cc_all)
                metrics_scores['cc'] = cc
                if return_raw:
                    cc_all_raw = np.full([Y.shape[1]], np.nan)
                    cc_all_raw[validix2] = cc_all
                    metrics_scores['ccraw'] = cc_all_raw
            elif m == 'r2':
                metrics_scores['r2weight'] = sklearn.metrics.r2_score(Y, Y_pred, multioutput='variance_weighted')
                metrics_scores['r2uniform'] = sklearn.metrics.r2_score(Y, Y_pred, multioutput='uniform_average')
                if return_raw: metrics_scores['r2raw'] = sklearn.metrics.r2_score(Y, Y_pred, multioutput='raw_values')

        return metrics_scores, Y_pred
