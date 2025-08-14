import numpy as np
from numpy.linalg import inv
import sklearn.metrics

class LinearRRR:
    def __init__(self, bias=True, rank='max', regList=[], zeromeanY=True, normstdY=True, verbose=False):
        """
        Reduced rank regression with separate L2 regularization weights for each predictor.
        :param bias: bool, whether to include a bias term.
        :param rank: 'max' or int, rank constraint on the regression coefficient matrix. If 'max', rank will not be constrained.
        :param regList: list of floats (optional). regList[i] specifies the L2 regularization weight applied to the i-th predictor. If empty, no L2 regularization will be used.
        :param zeromeanY: bool, if True, will subtract the mean of responses before fitting
        :param normstdY: bool, if True, will divide the responses by std before fitting
        :param verbose: bool, if True, self.fit will return intermediate components of weight matrix.
        """
        self.regList = regList
        self.bias = bias
        self.rank = rank
        self.zeromeanY = zeromeanY
        self.normstdY = normstdY
        self.zeromeanXs = []  # if True, will subtract the mean of the corr. predictor before fitting
        self.normstdXs = []  # if True, will normalize the std of corr. predictor before fitting
        self.regs = None
        self.meanXs = None
        self.stdXs = None
        self.meanY = None
        self.stdY = None
        self.weight = None
        self.verbose = verbose

    def fit(self, Xs, Y, zeromeanXs, normstdXs):
        """
        Fits the regression coefficient matrix.
        :param Xs: list of NumPy matrices, input to regression. Each matrix is one predictor; all matrices should have the same # of rows (i.e. same # of samples/observations).
        :param Y: NumPy matrix, regression targets. Should have the same # of rows as each predictor in Xs.
        :param zeromeanXs: binary list. If zeromeanXs[i]==1, subtract mean from the i-th predictor.
        :param normstdXs: binary list. If normstdXs[i]==1, divide the i-th predictor by its std.
        :return: if self.verbose is True, return intermediate components of weight matrix. Otherwise none.
        """
        assert (len(Xs) == len(self.regList)) or (len(self.regList) == 0), 'regularization shape mismatch'

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
            regs = np.append(regs, self.regList[i]*np.ones(X.shape[1]))

        # check std
        zerostdix = np.argwhere(stdXs < np.spacing(1))
        if len(zerostdix) > 0:
            stdXs[zerostdix.ravel()] = 1  # if std = 0 (often mean = 0 too), set it to 1 (i.e. not divide by std)

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
                # self.stdY = np.mean(Y, axis=0)
            Y = np.divide(Y, np.tile(self.stdY, (Y.shape[0], 1)))

        # make sure all data is valid
        cont_flag = 1
        if not np.sum(~np.isfinite(Xall)) == 0:
            print('invalid value found in X_train')
            cont_flag = 0
        assert cont_flag, 'invalid value found in data, cannnot proceed'

        if self.bias:  # add a column of ones for the bias term
            Xall = np.c_[np.ones([Xall.shape[0], 1]), Xall]
            if len(self.regList) > 0:
                Lbda = np.diag(self.regs)
                Lbda[0, 0] = 0  # do not apply regularization to bias
        else:
            if len(self.regList) > 0:
                Lbda = np.diag(self.regs)

        max_rank = np.min([Xall.shape[0], Xall.shape[1], Y.shape[1]])
        if self.rank == 'max':
            self.rank = max_rank
        assert 0 <= self.rank <= max_rank, 'Rank out of possible range'

        if len(self.regList) > 0:
            W_ols = inv(Xall.T @ Xall + Lbda) @ Xall.T @ Y  # least squares solution
        else:
            W_ols = inv(Xall.T @ Xall) @ Xall.T @ Y  # no regularization
        _u, _s, vh = np.linalg.svd(Y.T @ Xall @ W_ols)
        vh_r = vh[0:self.rank,:]
        Pr = vh_r.T @ vh_r
        self.weight = W_ols @ Pr
        if self.verbose:
            return W_ols, vh

    def predict(self, Xs, unnormY=False, weight=None):
        """
        Use model to make prediction.
        :param Xs: list of NumPy matrices, list of predictors.
        :param unnormY: bool, if True, unnormalize Y (i.e., multiple std and/or add mean back if self.zeromeanY and/or self.normstdY is True)
        :param weight: NumPy matrix (optional). If not None, will predict with the provided weight. If None, use the current trained model.
        :return: Ypred: predicted target matrix (NumPy matrix)
        """
        assert (self.weight is not None) or (weight is not None), 'fit the model before predicting or provide a weight matrix'

        Xall = np.hstack(Xs)
        Xall = Xall - np.tile(self.meanXs, (Xall.shape[0], 1))
        Xall = np.divide(Xall, np.tile(self.stdXs, (Xall.shape[0], 1)))

        # make sure all data is valid
        cont_flag = 1
        if not np.sum(~np.isfinite(Xall)) == 0:
            print('invalid value found in X_test')
            cont_flag = 0
        assert cont_flag, 'invalid value found in data, cannnot proceed'

        if self.bias:
            Xall = np.c_[np.ones([Xall.shape[0], 1]), Xall]

        if weight is not None:  # if a new weight is provided, then use it instead of the model weight
            Ypred = Xall @ weight
        else:
            Ypred = Xall @ self.weight

        if unnormY:
            if self.normstdY:
                Ypred = np.multiply(Ypred, np.tile(self.stdY, (Ypred.shape[0], 1)))
            if self.zeromeanY:
                Ypred = Ypred + np.tile(self.meanY, (Ypred.shape[0], 1))

        return Ypred

    def evaluate(self, Xs, Y, Ypred=None, unnormY=False, weight=None, metrics=['mse']):
        """
        Evaluate model prediction.
        :param Xs: list of NumPy matrices, list of predictors.
        :param Y: NumPy matrix, prediction targets. Should have the same # of rows as each predictor in Xs.
        :param Ypred: NumPy matrix (optional), predicted targets. If None, will call self.predict to generate prediction first.
        :param unnormY: bool, if True, unnormalize Y (i.e., multiple std and/or add mean back if self.zeromeanY and/or self.normstdY is True)
        :param weight: NumPy matrix (optional). If not None, will predict with the provided weight. If None, use the current trained model.
        :param metrics: list of str, metrics used to evaluate prediction.
                        Supported: 'mse' (mean squared error),
                                   'd2' (fraction of Poisson deviance explained; negative values in Ypred will be set to 0),
                                   'cc' (Pearson correlation coefficient),
                                   'r2' (R squared).
        :return: metrics_scores: dictionary of scores
        """
        if Ypred is None:
            Ypred = self.predict(Xs, unnormY=unnormY, weight=weight)

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
                cc_all_raw = np.full([Y.shape[1]], np.nan)
                cc_all_raw[validix2] = cc_all
                metrics_scores['ccraw'] = cc_all_raw

            elif m == 'r2':
                metrics_scores['r2weight'] = sklearn.metrics.r2_score(Y, Ypred, multioutput='variance_weighted')
                metrics_scores['r2uniform'] = sklearn.metrics.r2_score(Y, Ypred, multioutput='uniform_average')
            elif m == 'd2':
                Ypred_pos = np.clip(Ypred, a_min=0, a_max=None)  # prediction has to be positive to compute d2
                eps = np.spacing(1)  # add small number to avoid 0
                d2_all = np.array([sklearn.metrics.d2_tweedie_score(Y[:, i], Ypred_pos[:, i] + eps, power=1) for i in validix])
                d2uniform = np.average(d2_all)
                Y_null_pred = np.tile(np.mean(Y, axis=0), (Y.shape[0], 1))
                null_poisson_deviance = np.array([sklearn.metrics.mean_poisson_deviance(Y[:, i], Y_null_pred[:, i] + eps) for i in validix])
                d2nullweight = np.average(d2_all, weights=null_poisson_deviance)
                d2varweight = np.average(d2_all, weights=np.var(Y[:, validix], axis=0, ddof=1))
                metrics_scores['d2uniform'] = d2uniform
                metrics_scores['d2nullweight'] = d2nullweight
                metrics_scores['d2varweight'] = d2varweight
                d2_all_raw = np.full([Y.shape[1]], np.nan)
                d2_all_raw[validix] = d2_all
                metrics_scores['d2raw'] = d2_all_raw

        return metrics_scores

