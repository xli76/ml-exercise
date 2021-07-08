import mxnet as mx
from mxnet import gluon, nd, npx
from mxnet.gluon import nn
from mxnet.gluon.nn import activations
from mxnet.test_utils import create_vector

mx.random.seed(42)
# npx.set_np()  # Change MXNet to the numpy-like mode.

class MLP(nn.Block):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential()
        self.mlp.add(
            nn.Dense(units=128, activation='relu'),
            nn.Dense(units=64, activation='relu'),
            nn.Dense(units=10)
        )

    def forward(self, x):
        return self.mlp(x)

class LeNet(nn.Block):
    def __init__(self):
        super().__init__()
        self.lenet = nn.Sequential()
        # channel first, input shape (1, 28, 28)
        self.lenet.add(
            nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
            nn.MaxPool2D(pool_size=(2, 2)),
            nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
            nn.MaxPool2D(pool_size=(2, 2)),
            nn.Dense(units=120, activation='relu'),
            nn.Dense(units=84, activation='relu'),
            nn.Dense(units=10)
        )

    def forward(self, x):
        return self.lenet(x)

def train(model, x, y, epochs=5, batch_size=40, ctx=mx.cpu()):
    import time
    tic = time.time()
    btic = time.time()
    initializer = mx.initializer.Xavier()
    model.initialize(initializer, ctx=ctx)
    trainer = gluon.Trainer(model.collect_params(), 'adam', {'learning_rate': 0.001})
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()

    train_dataset = gluon.data.dataset.ArrayDataset(x, y)
    train_loader = gluon.data.DataLoader(train_dataset, batch_size=batch_size)

    accuracy = mx.metric.Accuracy()
    log_interval = 200

    for epoch in range(epochs):
        accuracy.reset()
        for idx, batch in enumerate(train_loader):
            data = batch[0].copyto(ctx)
            label = batch[1].copyto(ctx)

            with mx.autograd.record():
                outputs = model(data)
                loss = loss_fn(outputs, label)
            mx.autograd.backward(loss)
            trainer.step(batch_size)
            accuracy.update([label], [outputs])

            if log_interval and (idx + 1) % log_interval == 0:
                _, acc = accuracy.get()
                print(f"""Epoch[{epoch + 1}] Batch[{idx + 1}] Speed: {batch_size / (time.time() - btic)} samples/sec \
                      batch loss = {loss.mean().asscalar()} | accuracy = {acc}""")
                btic = time.time()
    
    _, acc = accuracy.get()

    print(f"[Epoch {epoch + 1}] training: accuracy={acc}")
    print(f"[Epoch {epoch + 1}] time cost: {time.time() - tic}")

def eval(model, x, y, ctx=mx.cpu):
    acc = mx.metric.Accuracy()
    x = mx.nd.array(x, ctx=ctx)
    y = mx.nd.array(y, ctx=ctx)
    output = model(x)
    acc.update([y], [output])
    _, accuracy = acc.get()
    print(f'validation accuracy = {accuracy}')

import gzip, struct, os
import numpy as np

def get_mnist(path):
    """Download and load the MNIST dataset

    Returns
    -------
    dict
        A dict containing the data
    """
    def read_data(label_url, image_url):
        with gzip.open(label_url) as flbl:
            struct.unpack(">II", flbl.read(8))
            label = np.frombuffer(flbl.read(), dtype=np.int8)
        with gzip.open(image_url, 'rb') as fimg:
            _, _, rows, cols = struct.unpack(">IIII", fimg.read(16))
            image = np.frombuffer(fimg.read(), dtype=np.uint8).reshape(len(label), rows, cols)
            image = image.reshape(image.shape[0], 1, 28, 28).astype(np.float32)/255
        return (label, image)

    # changed to mxnet.io for more stable hosting
    # path = 'http://yann.lecun.com/exdb/mnist/'
    # path = 'http://data.mxnet.io/data/mnist/'
    (train_lbl, train_img) = read_data(
        os.path.join(path, 'train-labels-idx1-ubyte.gz'), 
        os.path.join(path, 'train-images-idx3-ubyte.gz'))
    (test_lbl, test_img) = read_data(
        os.path.join(path, 't10k-labels-idx1-ubyte.gz'), 
        os.path.join(path, 't10k-images-idx3-ubyte.gz'))
    return {'train_data':train_img, 'train_label':train_lbl,
            'test_data':test_img, 'test_label':test_lbl}

if __name__ == '__main__':
    mnist = get_mnist('data')
    x_train, y_train = mnist['train_data'], mnist['train_label']
    x_test, y_test = mnist['test_data'], mnist['test_label']
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # input = nd.random.normal(shape=(1, 1, 28, 28))
    # m = nn.Sequential()
    # m.add(nn.Conv2D(channels=6, kernel_size=5, activation='relu'),
    #       nn.MaxPool2D(pool_size=(2, 2)),
    #       nn.Conv2D(channels=16, kernel_size=5, activation='relu'),
    #       nn.MaxPool2D(pool_size=(2, 2)))
    # initializer = mx.initializer.Xavier()
    # m.initialize(initializer)
    # output = m(input)
    # print(output.shape)
    # exit()

    model = MLP()
    # model = LeNet()
    train(model, x_train, y_train, ctx=mx.cpu())
    eval(model, x_test, y_test, ctx=mx.cpu())
    # conv2d channel first
    # x_train = x_train.reshape(-1, 1, 28, 28)
    # x_test = x_test.reshape(-1, 1, 28, 28)