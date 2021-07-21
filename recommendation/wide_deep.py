import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental import preprocessing

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

    embedding_inputs = []
    embedding_cols = []
    oht_cols = []
    for col in categorical_columns:
        # 低维离散特征
        vocab = pd.unique(all_data[col])
        input_dim = len(vocab)
        if input_dim <= 5:
            oht = OneHotEncoder()
            all_data[col] = oht.fit_transform(all_data[[col]]).todense()
            oht_cols.append(col)
        else:
            # embedding_inputs.append((col, input_dim, int(input_dim*0.4)))
            lbe = LabelEncoder()
            all_data[col] = lbe.fit_transform(all_data[col])
            embedding_inputs.append((col, vocab, int(input_dim*0.4)))
            embedding_cols.append(col)
    
    # 连续变量 continues index
    # continus_idx = [train_data.columns.get_loc(c) for c in continus_columns if c in train_data]


    y = all_data['label']
    x = all_data.drop(['label'], axis=1)
    x = all_data
    
    train_size = len(train_data)
    
    # one-hot, 连续变量, embedding
    train_x = x[:train_size]
    train_x_oht = train_x[oht_cols].values
    train_x_cont = train_x[continus_columns].values
    train_x_embed = train_x[embedding_cols].values
    test_x = x[train_size:]
    test_x_oht = test_x[oht_cols].values
    test_x_cont = test_x[continus_columns].values
    test_x_embed = test_x[embedding_cols].values
    train_y, test_y = y[:train_size].values, y[train_size:].values
    # wide model : one-hot特征, 交叉特征 
    # deep model : 连续特征，id类

    # for col in continus_columns:
    #     ss = StandardScaler()
    #     train_x[col] = ss.fit_transform(train_x[[col]])
    #     test_x[col] = ss.transform(test_x[[col]])

    return (train_x_oht, train_x_cont, train_x_embed), train_y, (test_x_oht, test_x_cont, test_x_embed), test_y, embedding_inputs

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

def create_model(embedding_inputs):
    oht_input = keras.layers.Input(shape=(2,)) #(2, 5)
    conti_input = keras.layers.Input(shape=(5,))
    embed_input = keras.layers.Input(shape=(6,))
    embed_inputs = keras.layers.Lambda(lambda x: [x[:, i] for i in range(6)])(embed_input)
    embeds = []
    for i, (col, vocab, dim) in enumerate(embedding_inputs):
        input = embed_inputs[i]
        embed = keras.layers.Embedding(len(vocab), dim, input_length=1, name='embed-{}'.format(col))(input)
        embed = keras.layers.Flatten(name='flatten-{}'.format(col))(embed)
        embeds.append(embed)
    wide_input = keras.layers.concatenate([oht_input, conti_input])
    deep_input = keras.layers.concatenate(embeds+[conti_input])
    wide_net = keras.layers.Dense(units=16, activation='relu')
    deep_net = tf.keras.models.Sequential([
            tf.keras.layers.Dense(units=256, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(units=64, activation='relu')
        ])
    wide = wide_net(wide_input)
    deep = deep_net(deep_input)
    both = tf.keras.layers.concatenate([wide, deep])
    output = tf.keras.layers.Dense(units=1, activation='sigmoid')(both)
    model = keras.Model(inputs=[oht_input, conti_input, embed_input], outputs=output)
    return model

def train(model, x, y):
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x, y)

def eval(model, x, y):
    model.evaluate(x, y)

if __name__ == '__main__':
    train_x, train_y, test_x, test_y, embedding_inputs = load_dataset()
    (train_x_oht, train_x_cont, train_x_embed) = train_x
    (test_x_oht, test_x_cont, test_x_embed) = test_x
    
    model = create_model(embedding_inputs)
    train(model, [train_x_oht, train_x_cont, train_x_embed], train_y)
    eval(model, [test_x_oht, test_x_cont, test_x_embed], test_y)
