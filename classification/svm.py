import numpy as np
import logging

logger = logging.getLogger("ml exercise")

class SVM:
    def __init__(self, tol=1e-3, C=1, max_iter=100) -> None:
        self.tol = tol
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        self.n_samples, self.n_dim = X.shape
        self.w = np.zeros(self.n_dim)
        self.b = 0
        self.X = X
        self.y = y
        self.K = np.dot(self.X, self.X.T)
        self.alphas = np.zeros(self.n_samples)
        self.errors = np.zeros((self.n_samples, 2))

        iter = 0
        alpha_changed = 0
        examine_all = True
        score = self.score(X, y)
        logger.info("initial score {}".format(score))
        while iter < self.max_iter and (alpha_changed > 0 or examine_all):
            iter += 1
            alpha_changed = 0
            if iter % 20 == 0:
            # if 1 == 1:
                new_score = self.score(X, y)
                logger.info("iteration {}, current score {}".format(iter, new_score))
            if examine_all:
                for i in range(self.n_samples):
                    alpha_changed += self.examine_example(i)
            else:
                non_bounds = np.nonzero((self.alphas > 0) * (self.alphas < self.C))[0]
                for i in non_bounds:
                    alpha_changed += self.examine_example(i)
            if examine_all:
                examine_all = False
            elif alpha_changed == 0:
                examine_all = True

    def predict(self, X):
        y_pred = np.sign(self.decision_function(X, dual=True))
        return y_pred

    def decision_function(self, X, dual=False):
        # f(x) = w^Tx + b
        # f(x) = \sum_{i=1}^n \alpha_i y_i K(x_i, x) + b
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if dual:
            K = np.dot(self.X, X.T)
            return np.multiply(K.T, self.alphas * self.y).sum(axis=1) + self.b
        return np.dot(X, self.w) + self.b

    def compute_error(self, X, y):
        fx = self.decision_function(X, dual=True)
        return fx - y

    def update_error(self, i):
        Ei = self.compute_error(self.X[i], self.y[i])
        self.errors[i] = [1, Ei]

    def score(self, X, y):
        return self.accuracy(X, y)
        # return self.hinge_loss(X, y)

    def accuracy(self, X, y):
        y_pred =  self.predict(X)
        return (y == y_pred).mean()
        
    def hinge_loss(self, X, y):
        y_pred =  self.predict(X)
        print("accurate predictions:", (y == y_pred).sum())
        return np.mean(np.maximum(1 - y * y_pred, 0))

    def examine_example(self, i):
        y1 = self.y[i]
        E1 = self.compute_error(self.X[i], self.y[i])
        r1 = E1 * y1
        alpha2 = self.alphas[i]
        if (r1 < -self.tol and alpha2 < self.C) or (r1 > self.tol and alpha2 > 0):
            # idx = ((self.alphas == self.C) * (self.alphas == 0))
            j, _ = self.select_j(i, E1)
            return self.take_step(i, j)
            # if (idx.sum() > 1):
            #     i = self.select_i(j, E2)
            #     if self._smo(i, j):
            #         return 1
            # else:
            #     return 0
        else:
            return 0
    
    def select_j(self, i, Ei):
        self.errors[i] = [1, Ei]
        maxE = 0
        alpha_index = np.nonzero(self.errors[:, 0])[0]
        j, Ej = 0, 0
        if len(alpha_index) > 1:
            for k in alpha_index:
                if k == i:
                    continue
                Ek = self.compute_error(self.X[k], self.y[k])
                deletaE = abs(Ei - Ek)
                if deletaE > maxE:
                    maxE = deletaE
                    j = k
                    Ej = Ek
        else:
            j = i
            while j == i:
                j = np.random.randint(0, self.n_samples - 1)
            Ej = self.compute_error(self.X[j], self.y[j])
        return j, Ej


    def take_step(self, i, j):
        if i == j:
            return 0

        # Compute L, H
        # c = self.alphas[i] * self.y[i] + self.alphas[j] * self.y[j]
        # L1, H1 = 0, self.C
        # l = c * self.y[j]
        # h = (c - self.C * self.y[i]) * self.y[j]
        # if self.y[i] == self.y[j]:
        #     H1 = min(self.C, l)
        #     L1 = max(0, h)
        # else:
        #     H1 = min(self.C, h)
        #     L1 = max(0, l)
        
        # L, H = L1, H1
        if self.y[i] != self.y[j]:
            L = max(0, self.alphas[j] - self.alphas[i])
            H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
        else:
            L = max(0, self.alphas[j] + self.alphas[i] - self.C)
            H = min(self.C, self.alphas[j] + self.alphas[i])
        if L == H:
            return 0
        # assert(L1 == L and H1 == H)
        #  E1 = f(x1) - y1 E2 = f(x2) - y2
        #  eta = (K11 + K22 - 2K12)
        #  alpha2_new = alpha2_old + y2(E1 - E2) / eta
        E1 = self.compute_error(self.X[i], self.y[i])
        E2 = self.compute_error(self.X[j], self.y[j])
        K11 = self.K[i][i]
        K22 = self.K[j][j]
        K12 = self.K[i][j]
        eta = K11 + K22 - 2*K12

        # corner case
        if eta <= 0:
            return 0
        # logger.info("i : {} j : {} eta : {}, E1 : {}, E2 : {}".format(i, j, eta, E1, E2))
        alpha2 = self.alphas[j] + self.y[j] * (E1 - E2) / eta
        alpha2 = np.clip(alpha2, L, H)
        s = self.y[i] * self.y[j]
        alpha1 = self.alphas[i] + (self.alphas[j] - alpha2) * s

        # logger.info("alphas {} {}".format(alpha1, alpha2))

        dalpha1 = self.alphas[i] - alpha1
        dalpha2 = self.alphas[j] - alpha2

        # update alphas

        self.alphas[j] = alpha2
        self.update_error(j)
        
        if abs(dalpha2) < 1e-5:
            return 0

        self.alphas[i] = alpha1
        self.update_error(i)

        # update b
        b1 = -E1 + self.K[i][i]*self.y[i]*dalpha1 + self.K[i][j]*self.y[j]*dalpha2 + self.b
        b2 = -E2 + self.K[i][j]*self.y[i]*dalpha1 + self.K[j][j]*self.y[j]*dalpha2 + self.b
        if 0 < alpha1 and alpha1 < self.C:
            self.b = b1
        elif 0 < alpha2 and alpha2 < self.C:
            self.b = b2
        else:
            self.b = (b1 + b2) / 2
        
        # update w
        self.w = np.multiply(self.X.T, self.y * self.alphas).sum(axis=1)

        return 1