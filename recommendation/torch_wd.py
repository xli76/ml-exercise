import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm1d
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

from wide_deep import load_dataset

class WideDeep(nn.Module):
    def __init__(self, embedding_inputs):
        super(WideDeep, self).__init__()
        self.embedding_inputs = embedding_inputs
        self.wide = self.wide_model()
        self.deep = self.deep_model()
    
    def embeddings(self):
        embeds = []
        for col, vocab, dim in embedding_inputs:
            embed = nn.Sequential([
                nn.Embedding(len(vocab), dim, input_length=1, name='embed-{}'.format(col))(input),
                nn.Flatten(name='flatten-{}'.format(col))(embed)])
            embeds.append(embed)
        return embeds
    
    def wide_model(self):
        wide = nn.Sequential([nn.Linear(10, 16),
                            nn.ReLU()])
        return wide

    def deep_model(self):
        deep = nn.Sequential([
            nn.Linear(None,256),
            nn.ReLU(),
            # nn.BatchNorm1d(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU()
        ])
        return deep

    def forward(self, x):
        inputs = x[:7]
        embed_inputs = x[7:]
        embeds = [ ]
        for i, embed in enumerate(embed_inputs):
            embeds.append(self.embeddings[i](embed))
        wide = self.wide(inputs)
        deep_inputs = torch.cat((embeds, inputs[2:]))
        deep = self.deep()
        

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


import numpy as np

if __name__ == '__main__':
    train_x, train_y, test_x, test_y, embedding_inputs = load_dataset()
    
    train_x = np.hstack(list(train_x))
    test_x = np.hstack(list(test_x))

    train_data = TensorDataset((train_x, train_y))
    train_loader = DataLoader(train_data, batch_size=4)
    
    model = WideDeep(embedding_inputs)

    train(model, train_loader)
    # eval(model, test_x, test_y)
