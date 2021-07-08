import tensorflow as tf
from tensorflow import keras

class MLP(tf.keras.Model):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = tf.keras.models.Sequential([
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=128, activation='relu'),
            tf.keras.layers.Dense(units=64, activation='relu'),
            tf.keras.layers.Dense(units=10, activation='softmax')
        ])
    
    def call(self, x):
        return self.mlp(x)

class LeNet(tf.keras.Model):
    def __init__(self):
        super(LeNet, self).__init__()
        # data_format='channels_last'
        self.lenet = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(filters=6, kernel_size=(5, 5), input_shape=(28, 28, 1), data_format="channels_last"),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Conv2D(filters=16, kernel_size=(5, 5)),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=500, activation='relu'),
            tf.keras.layers.Dense(units=10, activation='softmax')
        ])

    def call(self, x):
        return self.lenet(x)
    

def train(model, x, y):
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    model.fit(x, y, epochs=5)

def eval(model, x, y):
    model.evaluate(x, y)
    

if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    (x_train, y_train),(x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    # model = MLP()
    # conv2d channel last
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    model = LeNet()
    train(model, x_train, y_train)
    eval(model, x_test, y_test)