import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
from sklearn.metrics import d2_tweedie_score
from sklearn.metrics import explained_variance_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_poisson_deviance

class GLM(nn.Module):
    def __init__(self, n_input, n_output, act='softplus', rank=0, loss='poisson', seed=-1, W0=None):
        super(GLM, self).__init__()
        if seed >= 0:
            torch.manual_seed(seed)
        self.act = act
        self.rank = rank
        self.loss = loss
        if self.rank == 0:  # full rank glm
            self.linear = nn.Linear(n_input, n_output)
            if W0 is not None:
                self.linear.weight.data = torch.from_numpy(W0)
                self.linear.bias.data.zero_()
        else:  # reduced-rank glm
            assert (rank <= min(n_input, n_output)), 'rank {} is invalid; rank must be less than {}'.format(rank, min(n_input, n_output))
            self.linear1 = nn.Linear(n_input, rank)
            self.linear2 = nn.Linear(rank, n_output)
            if W0 is not None:
                u, s, vh = np.linalg.svd(W0, full_matrices=False)
                self.linear1.weight.data = torch.from_numpy(vh[:rank]).float()
                self.linear2.weight.data = torch.from_numpy(u[:,:rank]).float()
                self.linear1.bias.data.zero_()
                self.linear2.bias.data.zero_()

    def forward(self, x):
        if self.rank == 0:
            if self.act == 'exp':
                if self.loss == 'poisson':
                    x = self.linear(x)
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
                    x = self.linear2(x)
                else:
                    x = torch.exp(self.linear2(x))
            elif self.act == 'relu':
                x = F.relu(self.linear2(x))
            elif self.act == 'softplus':
                x = F.softplus(self.linear2(x))
            elif self.act == 'softplus2':
                x = F.softplus(self.linear2(x))**2
            elif self.act == 'softplus3':
                x = F.softplus(self.linear2(x))**3
        return x

class SeparateGLM:
    def __init__(self, n_input, n_output, act='softplus', rank=0, loss='poisson', zeromeanXs=[], normstdXs=[], regList=[], seed=-1, W0=None):
        self.n_input = n_input
        self.n_output = n_output
        self.act = act
        self.rank = rank
        self.loss = loss.lower()
        self.regList = regList
        self.zeromeanXs= zeromeanXs
        self.normstdXs = normstdXs
        self.seed = seed
        self.meanXs = None
        self.stdXs = None
        # define model
        self.glm = GLM(n_input=n_input, n_output=n_output, act=act, rank=rank, loss=loss, seed=seed, W0=W0)
        # define loss
        if self.loss == 'poisson':
            if self.act == 'exp':
                self.loss_func = nn.PoissonNLLLoss(log_input=True, full=True, reduction='mean')
            else:
                self.loss_func = nn.PoissonNLLLoss(log_input=False, full=True, reduction='mean')
        elif self.loss == 'mse':
            self.loss_func = nn.MSELoss()

    def normalize(self, Xs):
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

            # check std
            zerostdix = np.argwhere(stdXs < np.spacing(1))
            if len(zerostdix) > 0:
                stdXs[zerostdix.ravel()] = 1  # if std = 0 (often mean = 0 too), set it to 1 (i.e. not divide by std)

            self.meanXs = meanXs
            self.stdXs = stdXs

        Xall = np.hstack(Xs)
        Xall = Xall - np.tile(self.meanXs, (Xall.shape[0], 1))
        Xall = np.divide(Xall, np.tile(self.stdXs, (Xall.shape[0], 1)))

        # make sure all data is valid
        cont_flag = 1
        if not np.sum(~np.isfinite(Xall)) == 0:
            print('invalid value found in X_train')
            cont_flag = 0
        assert cont_flag, 'invalid value found in data, cannnot proceed'

        return Xall

    def train(self, Xs_train, Y_train, Xs_val=None, Y_val=None,
              lr=1, maxepoch=100, max_iter=10, history_size=10, grad_tol=1e-4, line_search=1,
              shuffle=True, track=10, patience=10, loss_tol=1e-6, progfile=None):
        # set random seed if any
        if self.seed >= 0:
            rng = np.random.default_rng(seed=self.seed)
            torch.manual_seed(self.seed)
        else:
            rng = np.random.default_rng()

        # use gpu if possible
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('train:', device)

        X_train = torch.from_numpy(self.normalize(Xs_train)).float()
        Y_train = torch.from_numpy(Y_train).float()
        X_train = X_train.to(device)
        Y_train = Y_train.to(device)
        val = (Xs_val is not None)
        if val:
            X_val = torch.from_numpy(self.normalize(Xs_val)).float()
            Y_val = torch.from_numpy(Y_val).float()
            X_val = X_val.to(device)
            Y_val = Y_val.to(device)
        self.glm.to(device)

        # define optimization method
        line_search_fn = 'strong_wolfe' if line_search else None
        optimizer = optim.LBFGS(self.glm.parameters(), lr=lr, history_size=history_size, max_iter=max_iter,
                                line_search_fn=line_search_fn)

        # define regularization
        if self.regList is not None:
            cumdim = np.cumsum([Xs_train[i].shape[1] for i in range(len(Xs_train))])
            cumdim = np.insert(cumdim, 0, 0)

        train_loss_hist = []
        val_loss_hist = []
        grad_hist = []

        stop = False
        e = 0
        cnt = 0

        with torch.no_grad():
            Y_train_pred = self.glm(X_train)
            train_loss_hist.append(self.loss_func(Y_train_pred, Y_train).item())
            if val:
                Y_val_pred = self.glm(X_val)
                val_loss_hist.append(self.loss_func(Y_val_pred, Y_val).item())

        if progfile is not None:
            toprint = 'epoch {}: training loss {:.6f} \n'.format(e, train_loss_hist[-1])
            progfile.write(toprint)
        else:
            print('epoch {}: training loss \n'.format(e), train_loss_hist[-1])

        while not stop:
            e += 1
            grad_step = torch.zeros(4)

            if shuffle:
                shuffled_ixs = rng.permutation(X_train.shape[0])
                X_train = X_train[shuffled_ixs]
                Y_train = Y_train[shuffled_ixs]

            def closure():
                optimizer.zero_grad()
                Y_train_hat = self.glm(X_train)
                loss = self.loss_func(Y_train_hat, Y_train)
                if self.regList is not None:
                    for k in range(len(Xs_train)):
                        loss += self.regList[k] * torch.linalg.norm((self.glm.linear2.weight @ self.glm.linear1.weight[:, cumdim[k]:cumdim[k + 1]]))
                loss.backward()
                return loss

            optimizer.step(closure)
            loss = closure()

            for l, param in enumerate(self.glm.parameters()):
                # grad_step[l] += torch.linalg.norm(param.grad)
                grad_step[l] += torch.max(abs(param.grad))

            train_loss_hist.append(loss.item())
            grad_hist.append(grad_step.numpy())

            if val:
                with torch.no_grad():
                    Y_val_pred = self.glm(X_val)
                    val_loss = self.loss_func(Y_val_pred, Y_val).item()
                    val_loss_hist.append(val_loss)

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
                    toprint = '{} loss changes less than {} after {} epochs \n '.format(patience, loss_tol, e)
                    progfile.write(toprint)
                else:
                    print('{} loss changes less than {} after {} epochs \n '.format(patience, loss_tol, e))

            if e % track == 0:
                if progfile is not None:
                    toprint = ['{} epochs: train loss {:.6f} \n'.format(e, train_loss_hist[-1]),
                               '           gradient {}\n'.format(grad_hist[-1])]
                    progfile.writelines(toprint)
                    progfile.flush()
                else:
                    print('epoch {}: training loss'.format(e), train_loss_hist[-1])
                    print('          gradient {}'.format(grad_hist[-1]))

            if e == maxepoch:
                stop = True
                if progfile is not None:
                    toprint = 'reached maximum epoch {} \n'.format(maxepoch)
                    progfile.write(toprint)
                else:
                    print('reached maximum epoch', maxepoch)

        return train_loss_hist, val_loss_hist, grad_hist

    def predict(self, Xs, modelfile=None):
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

        with torch.no_grad():
            Y_pred = self.glm(torch.from_numpy(self.normalize(Xs)).float()).cpu().numpy()

        if (self.loss == 'poisson') and (self.act == 'exp'):
            Y_pred = np.exp(Y_pred)

        return Y_pred

    def eval(self, Xs, Y, Y_pred=None, modelfile=None, metrics=['PNLL']):
        if Y_pred is None:
            Y_pred = self.predict(Xs, modelfile=modelfile)

        metrics_scores = {}

        Y_std = np.std(Y, axis=0)
        zerostdix = np.argwhere(Y_std == 0)
        validix = np.arange(Y.shape[1])
        mask = np.ones(Y.shape[1], dtype=bool)
        if len(zerostdix) > 0:
            mask[zerostdix.ravel()] = False
        validix = validix[mask]

        for m in metrics:
            if m == 'PNLL':
                loss_func = nn.PoissonNLLLoss(log_input=False, full=True, reduction='mean')
                test_loss = loss_func(torch.from_numpy(Y_pred).float(), torch.from_numpy(Y).float()).item()
                metrics_scores['PNLL'] = test_loss
            elif m == 'MSE':
                loss_func = nn.MSELoss()
                test_loss = loss_func(torch.from_numpy(Y_pred).float(), torch.from_numpy(Y).float()).item()
                metrics_scores['MSE'] = test_loss
            elif m == 'd2':
                eps = np.spacing(1)
                d2_all = np.array([d2_tweedie_score(Y[:, i], Y_pred[:, i] + eps, power=1) for i in validix])
                d2uniform = np.average(d2_all)
                Y_null_pred = np.tile(np.mean(Y, axis=0), (Y.shape[0], 1))
                null_poisson_deviance = np.array(
                    [mean_poisson_deviance(Y[:, i], Y_null_pred[:, i] + eps) for i in validix])
                d2nullweight = np.average(d2_all, weights=null_poisson_deviance)
                d2varweight = np.average(d2_all, weights=np.var(Y[:, validix], axis=0, ddof=1))
                metrics_scores['d2uniform'] = d2uniform
                metrics_scores['d2nullweight'] = d2nullweight
                metrics_scores['d2varweight'] = d2varweight
                d2_all_raw = np.full([Y.shape[1]], np.nan)
                d2_all_raw[validix] = d2_all
                metrics_scores['d2raw'] = d2_all_raw
            elif m == 'cc':
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
                cc_all_raw = np.full([Y.shape[1]], np.nan)
                cc_all_raw[validix2] = cc_all
                metrics_scores['ccraw'] = cc_all_raw
            elif m == 'r2':
                metrics_scores['r2weight'] = r2_score(Y, Y_pred, multioutput='variance_weighted')
                metrics_scores['r2uniform'] = r2_score(Y, Y_pred, multioutput='uniform_average')
                metrics_scores['r2raw'] = r2_score(Y, Y_pred, multioutput='raw_values')

            elif m == 'ev':
                metrics_scores['evweight'] = explained_variance_score(Y, Y_pred, multioutput='variance_weighted')
                metrics_scores['evuniform'] = explained_variance_score(Y, Y_pred, multioutput='uniform_average')
                metrics_scores['evraw'] = explained_variance_score(Y, Y_pred, multioutput='raw_values')

        return metrics_scores, Y_pred










