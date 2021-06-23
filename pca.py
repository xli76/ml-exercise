import numpy as np

class PCA:
    def __init__(self, n_components) -> None:
        self.n_components = n_components

    def fit(self, X):
        self.n_samples, self.n_features = X.shape
        self.mean = np.mean(X, axis=0)
        nX = X - self.mean
        # cov matrix d x d matrix
        self.covariance = np.dot(nX.T, nX)# / (self.n_samples - 1)
        # ith column vector is the eigen vector for ith eigen value
        eig_val, eig_vec = np.linalg.eig(self.covariance)
        idx = np.argsort(eig_val)[::-1][:self.n_components]
        print(eig_val[idx])
        print(eig_vec[:, idx])
        # d x k matrix, top k eigen vector
        self.m = eig_vec[:, idx]
        return self


    def transform(self, X):
        # n x d, d x k -> n x k
        mean = np.mean(X, axis=0)
        return np.dot(X-mean, self.m)