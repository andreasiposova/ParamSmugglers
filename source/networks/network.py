import sys
import torch
import torch.nn.functional as F
from torch import nn, optim
from collections import OrderedDict

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




class MLP_Net(nn.Module):
    def __init__(self, input_size, layer_size, num_hidden_layers, dropout):
        super().__init__()

        hidden_sizes = [int(layer_size * input_size) for _ in range(num_hidden_layers)]
        self.activations = []
        self.dropout = nn.Dropout(dropout)
        self.fcs = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])] + \
                                 [nn.Linear(hidden_sizes[i-1], hidden_sizes[i]) for i in range(1, num_hidden_layers)] + \
                                 [nn.Linear(hidden_sizes[-1], 1)])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def register_hooks(self):
        for fc in self.fcs:
            fc.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        # Store the output (i.e., activation) in the activations list
        self.activations.append(output)

    def remove_low_activations(self, threshold):
        # Set activations below the threshold to zero
        for i in range(len(self.activations)):
            self.activations[i] = torch.where(self.activations[i] >= threshold, self.activations[i],
                                              torch.zeros_like(self.activations[i]))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs) - 1:
                x = self.relu(x)
                x = self.dropout(x)
            else:
                x = self.sigmoid(x)
        return x.mean(dim=1)

    def forward_act(self, x):
        activations = []
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs) - 1:
                x = self.relu(x)
                x = self.dropout(x)
            else:
                x = self.sigmoid(x)
            activations.append(x)
        return x.mean(dim=1), activations

    def save_activation(self, module, input, output):
        # Mask out the activations of not active neurons
        masked_output = output.clone()
        masked_output[masked_output < 0] = 0

        # Save the masked activations
        self.activations.append(masked_output)

        # Remove the connections of not active neurons
        output[output < 0] = 0

    def penalty(self, s, params, lambda_s):
        #s is the secret vector, lambda_s is the penalty magnitude
        # Loop through all the weights of the model (to penalize all of them)
        # s is the secret vector (dictionary), lambda_s is the penalty magnitude
        total_penalty = 0
        targets = OrderedDict(s)
        for name, param in params.items():
            if name in targets:
                 #Extract the corresponding tensor from s
                s_tensor = s[name]
                # Ensure the shapes are compatible
                if s_tensor.shape == param.shape:
                    #penalty_for_param = torch.sum(torch.abs(torch.clamp(-param * s_tensor, min=0)))
                    constraint = -1 * s_tensor * param
                    size = param.size()
                    penalty_for_param = torch.abs(torch.where(constraint > 0, constraint, torch.zeros(size)))
                    total_penalty += torch.mean(penalty_for_param)
                else:
                    raise ValueError(f"Shape mismatch for parameter {name}: {param.shape} vs {s_tensor.shape}")
            else:
                raise KeyError(f"Parameter {name} not found in secret vector s")
        return total_penalty * lambda_s


class MLP_Net_x(nn.Module):
    def __init__(self, input_size, layer_size, num_hidden_layers, dropout):
        super().__init__()

        hidden_sizes = [int(layer_size) for _ in range(num_hidden_layers)]
        self.dropout = nn.Dropout(dropout)
        self.fcs = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])] + \
                                 [nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]) for i in
                                  range(1, num_hidden_layers)] + \
                                 [nn.Linear(hidden_sizes[-1], 1)])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def register_hooks(self):
        for fc in self.fcs:
            fc.register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        self.activations.append(output)

    def remove_low_activations(self, threshold):
        for i in range(len(self.activations)):
            self.activations[i] = torch.where(self.activations[i] >= threshold, self.activations[i],
                                              torch.zeros_like(self.activations[i]))

    def forward_act(self, x):
        activations = []
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs) - 1:
                x = self.relu(x)
                x = self.dropout(x)
            else:
                x = self.sigmoid(x)
            activations.append(x)
        return x.mean(dim=1), activations

    def forward(self, x):
        activations = []
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < len(self.fcs) - 1:
                x = self.relu(x)
                x = self.dropout(x)
            else:
                x = self.sigmoid(x)
            activations.append(x)
        return x.mean(dim=1)

    def get_activations(self, x):
        # Get activations per layer
        _, activations = self.forward(x)
        return activations

    def save_activation(self, module, input, output):
        masked_output = output.clone()
        masked_output[masked_output < 0] = 0
        self.activations.append(masked_output)
        output[output < 0] = 0


def build_optimizer(network, optimizer, learning_rate, weight_decay):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9, nesterov=True)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def build_network(input_size, m1, m2, m3, m4, dropout):
    model = Net(input_size, m1, m2, m3, m4, dropout)
    return model

def build_mlp(input_size, layer_size, num_hidden_layers, dropout):
    model = MLP_Net(input_size, layer_size, num_hidden_layers, dropout)
    return model