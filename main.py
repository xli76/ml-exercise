import logging
from matplotlib import pyplot as plt 
from data_utils import *
from clustering.kmeans import KMeans
from regression.linear_regression import LinearRegression
from classification.logistic_regression import LogisticRegression
from classification.svm import SVM
from pca import PCA

FORMAT = '%(asctime)s[%(levelname)s]:%(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)

logger = logging.getLogger("ml exercise")

def run_dimension_reduction():
    X = np.array([[-1, 1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    pca = PCA(n_components=1)
    pca.fit(X)
    print(pca.transform(X))

def run_clustering():
    centers = [(-5, -5), (0, 0), (5, 5)]
    X, y = generate_clustering_data(3, centers)
    plot_clustering(X, y, centers)
    # init = [[c[0], c[1]] for c in centers]
    kmeans = KMeans(3)
    kmeans.fit(X)
    # y_pred = kmeans.predict(X)
    plot_clustering(X, kmeans.labels, kmeans.centers)

def run_regression():
    X, y = generate_regression_data(200, [4, 3])
    train_X, train_y, test_X, test_y = train_test_split(X, y)
    lr = LinearRegression(max_iter=500)
    lr.fit(train_X, train_y)
    print("training score {}", lr.score(train_X, train_y))
    pred_y = lr.predict(test_X)
    test_score = lr.score(test_X, test_y)
    print("test score {}", test_score)
    plot_regression(test_X, pred_y)

def run_classification():
    X, y = generate_classification_data(500, [[0.6, 0.4], [1.8, 1.8]])
    # plot_classification(X, y)
    train_X, train_y, test_X, test_y = train_test_split(X, y)
    lr = LogisticRegression(max_iter=500)
    lr.fit(train_X, train_y)
    print("training score {}", lr.score(train_X, train_y))
    coef = lr.coef_
    b, k = -coef[:2] / coef[2]
    print(coef, k, b)
    plot_hyperplane(k, b, train_X, train_y)
    # pred_y = lr.predict_proba(test_X)
    test_score = lr.score(test_X, test_y)
    print("test score {}", test_score)

def run_svm():
    X, y = generate_classification_data(100, [[0.6, 0.4], [1.8, 1.8]])
    # plot_classification(X, y)
    y[y == 0] = -1
    train_X, train_y, test_X, test_y = train_test_split(X, y)
    svm = SVM()
    svm.fit(train_X, train_y)
    print("training score {}", svm.score(train_X, train_y))
    w = svm.w
    k = -w[0] / w[1]
    b = -svm.b / w[1]
    print(w, k, b)
    plot_hyperplane(k, b, train_X, train_y)


from data_utils import load_dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

def recommendation_lr():
    train_x, train_y, test_x, test_y = load_dataset()
    clf = LogisticRegression(max_iter=200)
    clf.fit(train_x, train_y)
    print('lr score', clf.score(test_x, test_y))

def recommendation_gbdt():
    train_x, train_y, test_x, test_y = load_dataset()
    clf = GradientBoostingClassifier()
    clf.fit(train_x, train_y)
    print('gbdt score', clf.score(test_x, test_y))

class GBDTLR():
    def __init__(self) -> None:
        self.continus_columns = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
        self.gbdt = GradientBoostingClassifier()
        self.lr = LogisticRegression()

    def fit(self, X, y):
        input_x = self.preprocess(X)
        self.gbdt.fit(input_x, y)
        leaf_values = self.gbdt.apply(input_x)[:, :, 0]
        self.lr.fit(leaf_values, y)
        return self

    def predict(self, X):
        return self.lr.predict(self.transform(X))

    def preprocess(self, X):
        # return X[self.continus_columns]
        return X

    def transform(self, X):
        input_x = self.preprocess(X)
        leaf_values = self.gbdt.apply(input_x)[:, :, 0]
        return leaf_values

    def score(self, X, y):
        return self.lr.score(self.transform(X), y)
        

def recommendation_gbdt_lr():
    train_x, train_y, test_x, test_y = load_dataset()
    clf = GBDTLR()
    clf.fit(train_x, train_y)
    print('gbdt score', clf.gbdt.score(clf.preprocess(train_x), train_y))
    print('gbdt + lr score', clf.score(test_x, test_y))
    
if __name__ == '__main__':
    # print("hello world")
    # run_clustering()
    # run_regression()
    # run_classification()
    # run_svm()
    # recommendation_lr()
    # recommendation_gbdt()
    recommendation_gbdt_lr()