import numpy as np
import logging

logger = logging.getLogger("ml exercise")

class LogisticRegression:
    def __init__(self, max_iter=100) -> None:
        self.max_iter = max_iter
        self.learning_rate = 1e-3

    def fit(self, X, y):
        self.n_samples = X.shape[0]
        self.n_dim = X.shape[1]
        self.coef_ = np.zeros((self.n_dim + 1, 1))
        iter = 0
        score = self.score(X, y)
        logger.info("initial score {}".format(score))
        while iter < self.max_iter:
            iter += 1
            self._grad_descent(X, y)
            new_score = self.score(X, y)
            if iter % 20 == 0:
                logger.info("iteration {}, current score {}".format(iter, new_score))
            # early stop
            if score - new_score < 1e-10:
                break
            score = new_score

        return self

    def predict(self, X):
        """
        postive : g > 0, prob > 0.5
        negative : g < 0, prob < 0.5
        """
        return self.decision_function(X) > 0

    def predict_proba(self, X):
        """
        probability of being positive
        """
        return self.sigmoid(self.decision_function(X)).ravel()

    def decision_function(self, X):
        X = self.dummy_X(X)
        return np.dot(X, self.coef_)

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))

    def score(self, X, y):
        """
        cross entropy loss
        """
        y_pred = self.predict_proba(X)
        log_loss = -(y * np.log(y_pred) + (1-y) * np.log(1-y_pred))
        return np.sum(log_loss) / y.shape[0]

    def dummy_X(self, X):
        dummy_col = np.ones((X.shape[0], 1))
        return np.hstack([dummy_col, X])

    def _grad_descent(self, X, y):
        grad_coef_ = self._grad(X, y)
        self.coef_ += self.learning_rate * grad_coef_

    def _grad(self, X, y):
        y_pred = self.predict_proba(X)
        r = y - y_pred
        X = self.dummy_X(X)
        grad = np.dot(X.T, r)
        if grad.ndim == 1:
            grad = grad.reshape(-1, 1)
        return grad