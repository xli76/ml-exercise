import numpy as np
import logging

logger = logging.getLogger("ml exercise")

class LinearRegression:
    def __init__(self, max_iter=500) -> None:
        self.max_iter = max_iter
        self.learning_rate = 1e-3

    def fit(self, X, y):
        """
        fit the parameters for training data (X, y)
        X : n * d size data matrix
        y : n * 1 size label vector
        coef_ : (d+1) * 1 vector
        # coeff_ : d * 1 vector
        # intercept_ : scalar
        """
        assert(X.shape[0] == y.shape[0])
        self.n_sample = X.shape[0]
        self.dim = X.shape[1]
        self.coef_ = np.zeros((self.dim + 1, 1))
        iter = 0
        score = self.score(X, y)
        logger.info("initial score {}".format(score))
        while iter <= self.max_iter:
            iter += 1
            self._grad_descent(X, y)
            new_score = self.score(X, y)
            if iter % 20 == 0:
                logger.info("iteration {}, current score {}".format(iter, new_score))
            # early stop
            if score - new_score < 1e-8:
                break
            score = new_score
        logger.info("finished training coeff : {}".format(self.coef_))
        return self

    def predict(self, X):
        """
        fit the parameters for training data (X, y)
        """
        y_pred = self.descision_function(X)
        # logger.info("input {}, output {}, coeff {}".format(X[0], y_pred[0], self.coef_))
        return y_pred.ravel()
    
    def descision_function(self, X):
        X = self.dummy_X(X)
        return np.dot(X, self.coef_)

    def score(self, X, y):
        y_pred = self.predict(X)
        return np.sum(np.square(y - y_pred)) / self.n_sample

    def _grad_descent(self, X, y):
        """
        one step gradient descent
        """
        grad_coef_ = self._grad(X, y)
        self.coef_ += self.learning_rate * grad_coef_

    def dummy_X(self, X):
        dummy_col = np.ones((X.shape[0], 1))
        return np.hstack([dummy_col, X])

    def _grad(self, X, y):
        y_pred = self.predict(X)
        r = y - y_pred
        X = self.dummy_X(X)
        return np.dot(X.T, r).reshape(-1, 1)