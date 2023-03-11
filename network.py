import torch.nn.functional as F
from torch import nn, optim


class Net(nn.Module):
    def __init__(self, input_size, m1, m2, m3, m4, dropout):
        super().__init__()
        hidden_size1 = m1 * input_size
        hidden_size2 = m2 * input_size
        hidden_size3 = m3 * input_size
        hidden_size4 = m4 * input_size

        super(Net, self).__init__()

        self.dropout = dropout
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size3)
        self.fc4 = nn.Linear(hidden_size3, hidden_size4)
        self.fc5 = nn.Linear(hidden_size4, 1)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.bn3 = nn.BatchNorm1d(hidden_size3)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        x = self.sigmoid(x)
        return x.mean(dim=1) #view(-1)

def build_optimizer(network, optimizer, learning_rate, weight_decay):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def build_network(input_size, m1, m2, m3, m4, dropout):
    model = Net(input_size, m1, m2, m3, m4, dropout)
    return model