import numpy as np
import logging

logger = logging.getLogger("ml exercise")

eps = 7/3 - 1 - 4/3

class Node:
    def __init__(self) -> None:
        # split feature for split node
        self.feature = None
        # split value for split node
        self.value = None
        # 左子树
        self.childrens = {}

class Tree:
    def __init__(self, root=None, max_depth=2) -> None:
        # 根结点
        if root is None:
            root = Node()
        self.root = root
        self.max_depth = max_depth

class DecisionTree:
    def __init__(self, criterion="entropy", max_depth=3) -> None:
        self.max_depth = max_depth
        self.criterion = criterion

    def fit(self, X, y):
        self.n_samples, self.n_dim = X.shape
        self.X, self.y = X, y
        self.features = ["feature {}".format(i) for i in range(self.n_dim)]
        self.tree = self.create_tree(X, y, self.features)
        return self

    def predict(self, X):
        if X.ndim == 2:
            print([x for x in X])
            return [self.classify(x) for x in X]
        return self.classify(X)
        
    def classify(self, X, features=None, tree=None):
        if tree is None:
            tree = self.tree
        if features is None:
            features = self.features
        if type(tree) is not dict:
            return tree
        
        print(tree.keys())
        (feature_name, feature_dict), = tree.items()
        # feature_name = list(tree.keys())[0]
        feature = features.index(feature_name)
        value = X[feature]
        # TODO : missing value
        sub_tree = feature_dict[value]
        print("{} : value {}, subtree {}".format(feature_name, value, sub_tree))


        return self.classify(X, features, sub_tree)

    def create_tree(self, X, y, features):
        # if len(np.unique(y)) == 1:
        #     return y[0]

        if X.shape[1] == 1:
            # majority voting
            values, counts = np.unique(y, return_counts=True)
            return values[np.argmax(counts)]

        # index for best feature
        best_feature = self.best_split(X, y)
        tree = {}
        # featues : feature names
        feature_name = features[best_feature]
        tree[feature_name] = {}
        sub_features = features[:]
        sub_features.pop(best_feature)

        feature_values = np.unique(X[:, best_feature])
        for value in feature_values:
            sub_X, sub_y = self.subset(X, y, best_feature, value)
            tree[feature_name][value] = self.create_tree(sub_X, sub_y, sub_features)
        
        return tree


    def best_split(self, X, y):
        columns = X.shape[1]
        best_entropy = 1e6
        best_feature = 0
        for feature in range(columns):
            entropy = self.split_entropy(X, y, feature)
            if entropy <= best_entropy:
                best_entropy = entropy
                best_feature = feature
        return best_feature

    # def information_gain(self, X, y, feature, threshold):
    #     return self.impurity(y) - self.split_impurity(X, y, feature, threshold)

    def split_entropy(self, X, y, feature):
        feature_values = np.unique(X[:, feature])
        subsets = [self.subset(X, y, feature, value) for value in feature_values]
        entropy = 0
        for _, labels in subsets:
            prob = len(labels) / len(y)
            entropy += prob * self.impurity(labels)
        return entropy

    def subset(self, X, y, feature, value):
        idx = (X[:, feature] == value)
        return np.delete(X[idx], feature, axis=1), y[idx]

    def impurity(self, y):
        counts = np.unique(y, return_counts=True)[1]
        p = counts / counts.sum()
        if self.criterion != "entropy":
            return self.gini(p)
        return self.cross_entropy(p)

    def cross_entropy(self, p):
        p = p[p>0]
        return (-p * np.log2(p+eps)).sum()

    def gini(self, p):
        return 1 - (p*p).sum()
    
def traverse_tree(root):
    print(root.keys())


if __name__ == "__main__":
    X = np.array([[0, 0, 0, 0],                       #数据集
            [0, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 0, 1, 2],
            [1, 0, 1, 2],
            [2, 0, 1, 2],
            [2, 0, 1, 1],
            [2, 1, 0, 1],
            [2, 1, 0, 2],
            [2, 0, 0, 0]])
    y = np.array([0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0])
    # X = np.array([[0, 0], [1, 1], [0, 1], [1, 0]])
    # y = np.array([0, 1, 0, 1])
    dt = DecisionTree()
    dt.fit(X, y)
    from pprint import pprint
    pprint(dt.tree)
    test_X = np.array([[0, 0, 0, 1],[2, 1, 0, 1],[2, 1, 0, 2],[0, 1, 1, 1]])
    print(test_X[-1])
    print(dt.predict(test_X[-1]))