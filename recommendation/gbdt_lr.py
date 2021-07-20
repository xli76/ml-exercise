import sys
import os
current_url = os.path.dirname(__file__)
parent_url = os.path.abspath(os.path.join(current_url, os.pardir))

sys.path.append(parent_url)

from data_utils import load_dataset
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

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


if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_dataset()
    clf = GBDTLR()
    clf.fit(train_x, train_y)
    print('gbdt score', clf.gbdt.score(clf.preprocess(train_x), train_y))
    print('gbdt + lr score', clf.score(test_x, test_y))
