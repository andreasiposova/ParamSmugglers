import torch
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


"""class MLP_Net(nn.Module):
    def __init__(self, input_size, m, ratio, num_hidden_layers, dropout):
        super().__init__()

        if ratio == 'equal':
            hidden_sizes = [int(m * input_size) for _ in range(num_hidden_layers)]
            hidden_sizes += [0] * (4 - num_hidden_layers)
        elif ratio == '4321':
            if num_hidden_layers == 4:
                hidden_sizes = [int(m * input_size * 4),
                                int(m * input_size * 3),
                                int(m * input_size * 2),
                                int(m * input_size * 1)]
            elif num_hidden_layers == 3:
                hidden_sizes = [int(m * input_size * 3),
                                int(m * input_size * 2),
                                int(m * input_size * 1),
                                0]
            elif num_hidden_layers == 2:
                hidden_sizes = [int(m * input_size * 2),
                                int(m * input_size * 1),
                                0,
                                0]
            elif num_hidden_layers == 1:
                hidden_sizes = [int(m * input_size * 1),
                                0,
                                0,
                                0]
            else:
                raise ValueError('Invalid number of layers')
        else:
            raise ValueError('Invalid ratio parameter')

        self.dropout = nn.Dropout(dropout)
        self.fcs = nn.ModuleList([nn.Linear(input_size, hidden_sizes[0])] + \
                                 [nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]) for i in range(1, num_hidden_layers)] + \
                                 [nn.Linear(hidden_sizes[-2], 1)])
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            if i == 0:
                x = fc(x)
            elif i < len(self.fcs) - 1 and self.fcs[i-1].out_features != 0:
                x = fc(x)
                x = self.relu(x)
                x = self.dropout(x)
            elif i == len(self.fcs) - 1 and self.fcs[i-1].out_features != 0:
                x = fc(x)
                x = self.sigmoid(x)
            else:
                continue
        return x.mean(dim=1)

"""

# Define your custom hook function


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
        # Register a forward hook on each layer
        # Register hooks for each fully connected layer


        # Define custom hook function

        # Register hooks for each fully connected layer
    def register_hooks(self):
        for fc in self.fcs:
            fc.register_forward_hook(self.forward_hook)

        #self.fcs.register_forward_hook(self.save_activation)
        #self.fc2.register_forward_hook(self.save_activation)
        #self.fc3.register_forward_hook(self.save_activation)

    def forward_hook(self, module, input, output):
        # Store the output (i.e., activation) in the activations list
        self.activations.append(output)
        # Print a message to check if the forward hook is being called
        #print("Forward hook called. Activation shape:", output.shape)

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

    def penalty(self, s, lambda_s):
        #s is the secret vector, lambda_s is the penalty magnitude
        # Loop through all the weights of the model (to penalize all of them)
        total_penalty = 0
        for param in self.parameters():
            penalty_for_param = torch.sum(torch.abs(torch.clamp(-param * s, min=0)))
            total_penalty += penalty_for_param
        return lambda_s * total_penalty


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
                              lr=learning_rate, momentum=0.9)
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