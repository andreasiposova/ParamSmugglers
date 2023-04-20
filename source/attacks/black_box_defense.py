from wandb.wandb_torch import torch

from source.networks.network import MLP_Net


def black_box_defense(network, X_train):
    # Get the activations of all layers
    activations = network.activations

    # Test the network on benign data
    y_pred = network(X_train)

    # Get the activations of all layers
    activations = network.activations

    # Remove the forward hooks to avoid affecting future computations
    network.fcs.register_forward_hook(None)

    return network, activations

