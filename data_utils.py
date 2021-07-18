import numpy as np

def generate_regression_data(n, coeff):
    """
    n = [n * d] * [d * 1]
    y = [n * 1]
    """
    dim = len(coeff) - 1
    X = np.random.rand(n, dim)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    y = np.dot(X, coeff[1:]) + coeff[0]
    return X, y

def generate_classification_data(n, centers, ratio=0.5):
    dim = len(centers[0])
    pivot = int(n * ratio)
    offset = np.random.normal(loc=(0,0), scale=0.1, size=(n,dim))
    X = np.vstack([centers[0] + offset[:pivot], centers[1] + offset[:pivot]])
    y = np.ones(n)
    y[pivot:] = 0
    return X, y

def generate_clustering_data(n_centers, centers=None, n_dim=2, cluster_size=30):
    # np.random.seed(0)
    if centers is not None:
        assert(n_centers == len(centers))
    else:
        centers = np.random.rand(n_centers, n_dim) * 5
    def generate_cluster(c):
        return c + np.random.randn(cluster_size, n_dim)
    X = np.vstack([generate_cluster(c) for c in centers])
    y = np.concatenate([[i] * cluster_size for i in range(n_centers)])
    return X, y

def train_test_split(X, y, ratio=0.7):
    n_samples = X.shape[0]
    n_train = int(n_samples * 0.7)
    random_num = np.random.rand(n_samples)
    idx = np.argsort(random_num)
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    train_X, test_X = X[train_idx], X[test_idx]
    train_y, test_y = y[train_idx], y[test_idx]
    return train_X, train_y, test_X, test_y

from matplotlib import pyplot as plt

def plot_regression(X, y):
    plt.scatter(X[:, 0], y, s=0.1)
    plt.xlim(-1, 2)
    plt.ylim(-2, 10)
    plt.show()

def plot_classification(X, y):
    pos_idx = (y == 1)
    plt.scatter(X[pos_idx, 0], X[pos_idx, 1], c='r')
    plt.scatter(X[~pos_idx, 0], X[~pos_idx, 1], c='g')
    plt.show()

def plot_clustering(X, y, centers=None):
    labels = np.unique(y)
    colors = ['r', 'g', 'b', 'y']
    color_cnt = len(colors)
    for i, c in enumerate(labels):
        idx = (y == c)
        plt.scatter(X[idx, 0], X[idx, 1], c=colors[i%color_cnt])
    if centers is not None:
        for i, center in enumerate(centers):
            plt.scatter(center[0], center[1], s=100, c=colors[i%color_cnt])
    plt.show()

def plot_hyperplane(w, b, X=None, y=None):
    xx = np.linspace(-2.5, 5)
    yy = w * xx + b
    plt.plot(xx, yy)
    if X is not None:
        pos_idx = (y == 1)
        plt.scatter(X[pos_idx, 0], X[pos_idx, 1], c='r')
        plt.scatter(X[~pos_idx, 0], X[~pos_idx, 1], c='g')
    plt.show()

import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_dataset():
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label', 'type']
    continus_columns = ['age','fnlwgt','education-num','capital-gain','capital-loss','hours-per-week']
    discrete_columns = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    train_data = pd.read_csv('data/adult.data', index_col=False, names=columns, delimiter=',')
    test_data = pd.read_csv('data/adult.test', index_col=False, names=columns, delimiter=',')
    train_data['type'] = 'train'
    test_data['type'] = 'test'
    train_data['label'] = train_data['label'].map(lambda x : 1 if x.strip() == '>50K' else 0)
    test_data['label'] = test_data['label'].map(lambda x : 1 if x.strip() == '>50K.' else 0)
    all_data = pd.concat([train_data, test_data], axis=0)
    all_data = pd.get_dummies(all_data, columns=discrete_columns)
    train_data = all_data[all_data['type']=='train'].drop(['type'], axis=1)
    test_data = all_data[all_data['type']=='test'].drop(['type'], axis=1)

    for col in continus_columns:
        ss = StandardScaler()
        train_data[col] = ss.fit_transform(train_data[[col]])
        test_data[col] = ss.transform(test_data[[col]])

    train_x = train_data.drop(['label'], axis=1)
    train_y = train_data['label']
    test_x = test_data.drop(['label'], axis=1)
    test_y = test_data['label']

    return train_x, train_y, test_x, test_y