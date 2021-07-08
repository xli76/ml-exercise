import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms
import torch.optim as optim

import gzip, os
import numpy as np

class MNISTDataset(Dataset):
    def __init__(self, folder, train=True, transform=None):
        self.folder = folder
        self.train = train
        (train_data, train_labels) = self._load_data()
        self.train_data = train_data
        self.train_labels = train_labels
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.train_data[index], int(self.train_labels[index])
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.train_data)

    def _load_data(self):
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte.gz"
        data = read_idx(os.path.join(self.folder, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte.gz"
        targets = read_idx(os.path.join(self.folder, label_file))

        return data, targets

import codecs

SN3_PASCALVINCENT_TYPEMAP = {
    8: (torch.uint8, np.uint8, np.uint8),
    9: (torch.int8, np.int8, np.int8),
    11: (torch.int16, np.dtype('>i2'), 'i2'),
    12: (torch.int32, np.dtype('>i4'), 'i4'),
    13: (torch.float32, np.dtype('>f4'), 'f4'),
    14: (torch.float64, np.dtype('>f8'), 'f8')
}

def get_int(b: bytes) -> int:
    return int(codecs.encode(b, 'hex'), 16)

def read_idx(path):
    """Read a SN3 file in "Pascal Vincent" format (Lush file 'libidx/idx-io.lsh').
       Argument may be a filename, compressed filename, or file object.
    """
    # read
    with gzip.open(path, "rb") as f:
        data = f.read()
    # parse
    magic = get_int(data[:4])
    nd = magic % 256
    ty = magic // 256
    assert 1 <= nd <= 3
    assert 8 <= ty <= 14
    m = SN3_PASCALVINCENT_TYPEMAP[ty]
    s = [get_int(data[4 * (i + 1): 4 * (i + 2)]) for i in range(nd)]
    parsed = np.frombuffer(data, dtype=m[1], offset=(4 * (nd + 1)))
    # return parsed.astype(m[2], copy=False).reshape(s)
    return torch.from_numpy(parsed.astype(m[2], copy=False)).view(*s)

        
# def read_file(file_path):
#     with gzip.open(file_path, 'rb') as fn:
#         data = np.frombuffer(fn.read(), np.uint8, offset=8)
#     return data


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.Linear(128, 64),
            nn.Linear(64, 10)
        )
    
    def forward(self, x):
        return self.mlp(x)

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.lenet = nn.Sequential(
            # 1 input image channel, 6 output channels, 5x5 square convolution
            nn.Conv2d(1, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2)),
            nn.Flatten(),
            nn.Linear(4*4*16, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )
    
    def forward(self, x):
        return self.lenet(x)

def train(model, data_loader, epochs=5):
    log_interval = 2000
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i, data in enumerate(data_loader, 0):
            inputs, labels = data

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

            if log_interval and (i+1) % log_interval == 0:
                print("interval")
                print('[%d, %5d] loss: %.3f accuracy: %.3f' %
                        (epoch + 1, i + 1, running_loss / log_interval, 100 * correct / total))
                running_loss = 0.0
    

def eval(model, data_loader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in data_loader:
            inputs, labels = data
            # calculate outputs by running images through the network
            outputs = model(inputs)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy: %.3f' % (100 * correct / total))


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.Lambda(lambda img : img.div(255.0).unsqueeze(0))]
    )
    train_data = MNISTDataset('data', train=True, transform=transform)
    test_data = MNISTDataset('data', train=False, transform=transform)
    
    train_data_loader = DataLoader(train_data, batch_size=20)
    test_data_loader = DataLoader(test_data, batch_size=20)
    
    # model = MLP()
    model = LeNet()
    train(model, train_data_loader)
    eval(model, test_data_loader)
    