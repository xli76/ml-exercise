import logging
from matplotlib.pyplot import scatter
import numpy as np
from scipy.spatial import distance

logger = logging.getLogger("ml exercise")

class KMeans:
    def __init__(self, n_clusters, init=None, n_init=10, max_iter=100) -> None:
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        if init is not None:
            self.centers = init
            self.n_init = 1
        else:
            self.n_init = n_init

    def fit(self, X):
        best_score = 1e10
        for i in range(self.n_init):
            logger.info("running {} times".format(i))
            score = self.run(X)
            if score < best_score:
                logger.info("current score {}, best score {}".format(score, best_score))
                best_score = score
                best_labels = self.labels
                best_centers = self.centers
        self.labels = best_labels
        self.centers = best_centers
        return self
    
    def run(self, X):
        iter = 0
        random_idx = np.random.choice(X.shape[0], self.n_clusters)
        self.centers = X[random_idx]
        self.labels = np.zeros(X.shape[0])
        # logger.info("random choose {} centers : {} {}".format(len(self.centers), random_idx, self.centers))
        score = self.score(X)
        while iter < self.max_iter:
            iter += 1
            self.update_label(X)
            self.update_centers(X)
            new_score = self.score(X)
            if iter % 20 == 0:
                # logger.info("centers : {}".format(self.centers))
                logger.info("iteration {}, current score {}".format(iter, new_score))
            if score - new_score < 1e-10:
                break
            score = new_score
        return score

    def score(self, X):
        y_pred = self.predict(X)
        pred_centers = [self.centers[c] for c in y_pred]
        sum_scores = ((((X - pred_centers) ** 2).sum(axis=1))**0.5).sum()
        return sum_scores / self.n_clusters

    def predict(self, X):
        dist = self.distance_to_centers(X)
        return np.argmin(dist, axis=1)

    def update_label(self, X):
        self.labels = self.predict(X)

    def update_centers(self, X):
        for i in range(self.n_clusters):
            idx = (self.labels == i)
            self.centers[i] = np.sum(X[idx], axis=0) / idx.sum()

    def distance_to_centers(self, X):
        return distance.cdist(X, self.centers)