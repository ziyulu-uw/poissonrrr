"""
Linear reduced-rank regression utilities.

This module provides `LinearRRR`, a NumPy-based reduced-rank regression model
with optional:
- predictor-specific L2 regularization,
- predictor normalization,
- response centering/scaling, and
- a bias term implemented via predictor augmentation.

The fitted model follows the standard linear-RRR recipe: compute a regularized
OLS solution, then project it onto a lower-rank subspace.
"""

import numpy as np
from numpy.linalg import inv
import sklearn.metrics


class LinearRRR:
    """Linear reduced-rank regression with optional blockwise L2 penalties."""

    def __init__(self, bias=True, rank='max', regList=None, zeromeanY=True, normstdY=True, verbose=False):
        """
        Initialize a linear reduced-rank regression model.

        Parameters
        ----------
        bias : bool, default=True
            Whether to include a bias term by augmenting the predictor matrix
            with a column of ones.
        rank : {'max'} or int, default='max'
            Rank constraint on the regression coefficient matrix. If `'max'`,
            the model uses the largest admissible rank.
        regList : list[float] or None, default=None
            Predictor-block-specific L2 penalties. `regList[i]` applies to the
            i-th predictor block in `Xs`. Passing `None` or `[]` disables L2
            regularization.
        zeromeanY : bool, default=True
            Whether to subtract the response mean before fitting.
        normstdY : bool, default=True
            Whether to divide the response by its feature-wise standard
            deviation before fitting.
        verbose : bool, default=False
            If True, `fit` returns intermediate matrices from the reduced-rank
            construction.
        """
        regList = [] if regList is None else regList
        self.regList = regList
        self.bias = bias
        self.rank = rank
        self.zeromeanY = zeromeanY
        self.normstdY = normstdY
        self.zeromeanXs = []  # Placeholders populated during fit with per-block mean-centering flags.
        self.normstdXs = []  # Placeholders populated during fit with per-block std-normalization flags.
        self.regs = None
        self.meanXs = None
        self.stdXs = None
        self.meanY = None
        self.stdY = None
        self.weight = None
        self.fit_rank = None
        self.verbose = verbose

    def fit(self, Xs, Y, zeromeanXs, normstdXs):
        """
        Fit the reduced-rank regression coefficient matrix.

        Parameters
        ----------
        Xs : list of np.ndarray
            Predictor blocks. Each block must be 2D and all blocks must share
            the same number of rows (samples).
        Y : np.ndarray
            Regression targets with the same number of rows as each predictor
            block in `Xs`.
        zeromeanXs : list
            Per-block mean-centering flags.
        normstdXs : list
            Per-block standard-deviation normalization flags.

        Returns
        -------
        tuple[np.ndarray, np.ndarray] or None
            If `self.verbose` is True, returns `(W_ols, vh)` where `W_ols` is
            the ridge regression solution and `vh` contains the right singular
            vectors used for the rank truncation. Otherwise returns None.

        Raises
        ------
        ValueError
            If regularization dimensions do not match the predictor blocks, if
            normalized predictors contain non-finite values, or if the
            requested rank is outside the admissible range.
        """
        if len(self.regList) not in (0, len(Xs)):
            raise ValueError('regularization shape mismatch')

        self.zeromeanXs = zeromeanXs
        self.normstdXs = normstdXs

        meanXs = np.array([])
        stdXs = np.array([])

        if self.bias:
            regs = np.array([0])
        else:
            regs = np.array([])

        for i, X in enumerate(Xs):  # prepare each predictor separately
            if zeromeanXs[i]:
                meanX = np.mean(X, axis=0)
            else:
                meanX = np.zeros(X.shape[1])

            if normstdXs[i]:
                stdX = np.std(X, axis=0)
            else:
                stdX = np.ones(X.shape[1])

            meanXs = np.append(meanXs, meanX)
            stdXs = np.append(stdXs, stdX)
            reg = self.regList[i] if self.regList else 0
            regs = np.append(regs, reg * np.ones(X.shape[1]))

        # Constant features get unit scaling so normalization does not divide by zero.
        zerostdix = np.argwhere(stdXs < np.spacing(1))
        if len(zerostdix) > 0:
            stdXs[zerostdix.ravel()] = 1

        self.meanXs = meanXs
        self.stdXs = stdXs
        self.regs = regs

        Xall = np.hstack(Xs)
        Xall = Xall - np.tile(self.meanXs, (Xall.shape[0], 1))
        Xall = np.divide(Xall, np.tile(self.stdXs, (Xall.shape[0], 1)))

        if self.zeromeanY:
            if self.meanY is None:
                self.meanY = np.mean(Y, axis=0)
            Y = Y - np.tile(self.meanY, (Y.shape[0], 1))

        if self.normstdY:
            if self.stdY is None:
                self.stdY = np.std(Y, axis=0)
            Y = np.divide(Y, np.tile(self.stdY, (Y.shape[0], 1)))

        if np.any(~np.isfinite(Xall)):
            raise ValueError('invalid value found in data, cannot proceed')

        if self.bias:  # add a column of ones for the bias term
            Xall = np.c_[np.ones([Xall.shape[0], 1]), Xall]
            if self.regList:
                Lbda = np.diag(self.regs)
                Lbda[0, 0] = 0  # do not apply regularization to bias
        else:
            if self.regList:
                Lbda = np.diag(self.regs)

        max_rank = np.min([Xall.shape[0], Xall.shape[1], Y.shape[1]])
        fit_rank = max_rank if self.rank == 'max' else self.rank
        if not 0 <= fit_rank <= max_rank:
            raise ValueError('Rank out of possible range')
        self.fit_rank = fit_rank

        if self.regList:
            W_ols = inv(Xall.T @ Xall + Lbda) @ Xall.T @ Y  # Ridge regression solution.
        else:
            W_ols = inv(Xall.T @ Xall) @ Xall.T @ Y  # no regularization
        # Linear RRR is obtained by projecting the OLS solution onto the top
        # singular subspace of Y^T X W_ols.
        _u, _s, vh = np.linalg.svd(Y.T @ Xall @ W_ols)
        vh_r = vh[0:fit_rank, :]
        Pr = vh_r.T @ vh_r
        self.weight = W_ols @ Pr
        if self.verbose:
            return W_ols, vh

    def predict(self, Xs, unnormY=False, weight=None):
        """
        Generate predictions from the fitted model or a provided weight matrix.

        Parameters
        ----------
        Xs : list of np.ndarray
            Predictor blocks.
        unnormY : bool, default=False
            If True, undo response normalization before returning predictions.
        weight : np.ndarray or None, default=None
            Optional regression weight matrix. If provided, use this matrix
            instead of `self.weight`.

        Returns
        -------
        np.ndarray
            Predicted target matrix.

        Raises
        ------
        ValueError
            If the model has not been fit and no weight matrix is provided, or
            if normalized predictors contain non-finite values.
        """
        if (self.weight is None) and (weight is None):
            raise ValueError('fit the model before predicting or provide a weight matrix')

        Xall = np.hstack(Xs)
        Xall = Xall - np.tile(self.meanXs, (Xall.shape[0], 1))
        Xall = np.divide(Xall, np.tile(self.stdXs, (Xall.shape[0], 1)))

        if np.any(~np.isfinite(Xall)):
            raise ValueError('invalid value found in data, cannot proceed')

        if self.bias:
            Xall = np.c_[np.ones([Xall.shape[0], 1]), Xall]

        if weight is not None:  # Predict with the provided weight instead of the fitted model.
            Ypred = Xall @ weight
        else:
            Ypred = Xall @ self.weight

        if unnormY:
            if self.normstdY:
                Ypred = np.multiply(Ypred, np.tile(self.stdY, (Ypred.shape[0], 1)))
            if self.zeromeanY:
                Ypred = Ypred + np.tile(self.meanY, (Ypred.shape[0], 1))

        return Ypred

    def evaluate(self, Xs, Y, Ypred=None, unnormY=False, weight=None, metrics=None, return_raw=False):
        """
        Evaluate model predictions with regression and Poisson-style metrics.

        Parameters
        ----------
        Xs : list of np.ndarray
            Predictor blocks.
        Y : np.ndarray
            Prediction targets with the same number of rows as each predictor
            block in `Xs`.
        Ypred : np.ndarray or None, default=None
            Optional precomputed predictions. If None, `self.predict(...)` is
            called first.
        unnormY : bool, default=False
            If True, undo response normalization before generating predictions.
        weight : np.ndarray or None, default=None
            Optional regression weight matrix used when `Ypred` is not
            provided.
        metrics : list[str] or None, default=None
            Metrics to compute. Supported values are:
            `'mse'`, `'d2'`, `'cc'`, and `'r2'`. If None, defaults to
            `['mse']`.
        return_raw : bool, default=False
            If True, include per-output raw arrays for metrics that support
            them (`ccraw` and `d2raw`).

        Returns
        -------
        dict
            Dictionary of evaluation scores. Depending on requested metrics,
            keys may include `mse`, `cc`, `ccraw`, `r2weight`, `r2uniform`,
            `d2uniform`, `d2nullweight`, `d2varweight`, and `d2raw`.
        """
        if Ypred is None:
            Ypred = self.predict(Xs, unnormY=unnormY, weight=weight)

        metrics = ['mse'] if metrics is None else metrics

        metrics_scores = {}

        # Filter out outputs with zero variance; metrics like d2 and cc are not
        # informative or stable on constant targets.
        Y_std = np.std(Y, axis=0)
        zerostdix = np.argwhere(Y_std == 0)
        validix = np.arange(Y.shape[1])
        mask = np.ones(Y.shape[1], dtype=bool)
        if len(zerostdix) > 0:
            mask[zerostdix.ravel()] = False
        validix = validix[mask]

        for m in metrics:
            if m == 'mse':
                metrics_scores['mse'] = sklearn.metrics.mean_squared_error(Y, Ypred)
            elif m == 'cc':
                # filter out neurons whose predicted spike train has std=0. cc will throw error on them.
                Ypred_std = np.std(Ypred, axis=0)
                zerostdix = np.argwhere(Ypred_std == 0)
                validix1 = np.arange(Ypred.shape[1])
                mask = np.ones(Ypred.shape[1], dtype=bool)
                if len(zerostdix) > 0:
                    mask[zerostdix.ravel()] = False
                validix1 = validix1[mask]
                validix2 = np.intersect1d(validix, validix1)

                cc_all = np.array([np.corrcoef(Y[:, i], Ypred[:, i])[0, 1] for i in validix2])
                cc = np.average(cc_all)
                metrics_scores['cc'] = cc
                if return_raw:
                    cc_all_raw = np.full([Y.shape[1]], np.nan)
                    cc_all_raw[validix2] = cc_all
                    metrics_scores['ccraw'] = cc_all_raw

            elif m == 'r2':
                metrics_scores['r2weight'] = sklearn.metrics.r2_score(Y, Ypred, multioutput='variance_weighted')
                metrics_scores['r2uniform'] = sklearn.metrics.r2_score(Y, Ypred, multioutput='uniform_average')
            elif m == 'd2':
                Ypred_pos = np.clip(Ypred, a_min=0, a_max=None)  # d2_tweedie_score requires non-negative mean predictions.
                eps = np.spacing(1)  # Keep predictions strictly positive for Poisson-based metrics.
                d2_all = np.array([sklearn.metrics.d2_tweedie_score(Y[:, i], Ypred_pos[:, i] + eps, power=1) for i in validix])
                d2uniform = np.average(d2_all)
                Y_null_pred = np.tile(np.mean(Y, axis=0), (Y.shape[0], 1))
                null_poisson_deviance = np.array([sklearn.metrics.mean_poisson_deviance(Y[:, i], Y_null_pred[:, i] + eps) for i in validix])
                d2nullweight = np.average(d2_all, weights=null_poisson_deviance)
                d2varweight = np.average(d2_all, weights=np.var(Y[:, validix], axis=0, ddof=1))
                metrics_scores['d2uniform'] = d2uniform
                metrics_scores['d2nullweight'] = d2nullweight
                metrics_scores['d2varweight'] = d2varweight
                if return_raw:
                    d2_all_raw = np.full([Y.shape[1]], np.nan)
                    d2_all_raw[validix] = d2_all
                    metrics_scores['d2raw'] = d2_all_raw

        return metrics_scores
