import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras.layers.normalization import BatchNormalization

def load_dataset():
    columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 
    'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'label']
    continus_columns = ['age','education-num','capital-gain','capital-loss','hours-per-week']
    categorical_columns = ['workclass','education','marital-status','occupation','relationship','race','sex','native-country']
    train_data = pd.read_csv('data/adult.data', index_col=False, names=columns, delimiter=',')
    test_data = pd.read_csv('data/adult.test', skiprows=1, index_col=False, names=columns, delimiter=',')

    train_data = train_data.dropna(how='any', axis=0)
    test_data = test_data.dropna(how='any', axis=0)

    train_data['label'] = train_data['label'].apply(lambda x : '>50k' in x).astype(int)
    test_data['label'] = test_data['label'].apply(lambda x : '>50k' in x).astype(int)

    all_data = pd.concat([train_data, test_data])

    # for col in categorical_columns:
    #     oht = OneHotEncoder()
    #     all_data[col] = oht.fit_transform(all_data[[col]])
    
    y = all_data['label'].values
    x = all_data.drop(['label'], axis=1)
    x = all_data[continus_columns].values
    
    train_size = len(train_data)
    train_x, train_y = x[:train_size].copy(), y[:train_size].copy()
    test_x, test_y = x[train_size:].copy(), y[train_size:].copy()

    # cross product 离散特征
    # wide model : 离散特征, 交叉特征， id类
    # deep model : 连续特征直接使用，离散特征需要先加emebdding层
    # for col in continus_columns:
    #     ss = StandardScaler()
    #     train_x[col] = ss.fit_transform(train_x[[col]])
    #     test_x[col] = ss.transform(test_x[[col]])

    return train_x, train_y, test_x, test_y

class WideDeepModel(tf.keras.Model):
    def __init__(self) -> None:
        super(WideDeepModel, self).__init__()
        self.wide = self.wide_model()
        # self.deep = self.deep_model()
        self.model = self.combined_model()

    def wide_model(self):
        wide = keras.layers.Dense(units=16, activation='relu')
        return wide

    def deep_model(self):
        tf.keras.layers.Embedding(input_dim=100, output_dim=10)
        deep = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=64, activation='relu')
        ])
        return deep

    def combined_model(self):
        model = tf.keras.layers.Dense(units=1, activation='sigmoid')
        return model

    def call(self, x):
        out = self.wide(x)
        out = self.model(out)
        return out

def train(model, x, y):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    # model.build(input_shape=(None, x.shape[1]))
    model.fit(x, y)
    # print(model.summary())

def eval(model, x, y):
    model.evaluate(x, y)

if __name__ == '__main__':
    train_x, train_y, test_x, test_y = load_dataset()
    wide_deep = WideDeepModel()
    # wide_deep = keras.Sequential([
    #     keras.layers.Dense(units=16, activation='relu'),
    #     keras.layers.Dense(units=1, activation='sigmoid')
    # ])
    train(wide_deep, train_x, train_y)
    eval(wide_deep, test_x, test_y)