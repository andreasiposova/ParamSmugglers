import copy

import torch.nn as nn
from wandb.wandb_torch import torch

from source.networks.network import MLP_Net

def prune_neurons(model, activations, threshold):
    pruned_model = copy.deepcopy(model)
    for i, (activation, fc) in enumerate(zip(activations[:-1], pruned_model.fcs[:-1])):
        neuron_mask = activation.mean(dim=0) > threshold
        pruned_weight = fc.weight.data[:, neuron_mask]
        pruned_bias = fc.bias.data[neuron_mask]
        pruned_fc = nn.Linear(pruned_weight.shape[1], pruned_weight.shape[0])
        pruned_fc.weight.data = pruned_weight
        pruned_fc.bias.data = pruned_bias
        pruned_model.fcs[i] = pruned_fc
    return pruned_model


def black_box_defense(network, X_train, threshold):
    # Get the activations of all layers
    activations = network.activations
    # Test the network on benign data
    output = network(X_train)
    activations_fp = network(X_train)
    # Remove low activations
    threshold = 0.1
    #network.remove_low_activations(threshold)

    # Remove the forward hooks to avoid affecting future computations
    #network.fcs.register_forward_hook(None)
    pruned_network = prune_neurons(network, activations, threshold)
    pruned_activations = pruned_network.activations

    return pruned_network



