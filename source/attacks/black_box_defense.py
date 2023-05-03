import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from wandb.wandb_torch import torch as wandb_torch

from source.networks.network import MLP_Net

from torch.nn.utils import prune

def prune_neurons_old(model, activations, amount):
    #pruned_model = copy.deepcopy(model)
    pruned_model = model
    for i, (activation, fc) in enumerate(zip(activations[:-1], pruned_model.fcs[:-1])):
        neuron_mask = activation.mean(dim=0) > amount
        pruned_weight = fc.weight.data[:, neuron_mask]
        pruned_bias = fc.bias.data[neuron_mask]
        pruned_fc = nn.Linear(pruned_weight.shape[1], pruned_weight.shape[0])
        #prune.l1_unstructured(fc, name='weight', amount=amount)
        pruned_fc.weight.data = pruned_weight
        pruned_fc.bias.data = pruned_bias
        pruned_model.fcs[i] = pruned_fc
        prune.remove(fc, name='weight')
    return pruned_model

def create_neuron_mask(activations, threshold):
    #activations_tensor = torch.stack(activations)
    activations_tensor = activations.detach()
    # Calculate the mean activation for each neuron
    mean_activation = activations.mean(dim=0)
    #max_activations, _ = torch.max(activations_tensor, dim=0)
    mean_activation = mean_activation.detach()
    # Sort the average activations and their corresponding indices
    sorted_activations, sorted_indices = torch.sort(mean_activation)

    # Select the indices corresponding to the top k active neurons
    k = len(mean_activation)*threshold
    top_k_indices = sorted_indices[-k:]

    # Create a boolean mask using the selected indices
    neuron_mask = torch.zeros_like(mean_activation, dtype=torch.bool)
    neuron_mask[top_k_indices] = True

    neuron_mask = mean_activation > threshold
    return neuron_mask


def prune_neurons(model, layer_index, neuron_mask):
    layer = model.fcs[layer_index]

    # Get the indices where the neuron_mask is True
    mask_indices = neuron_mask.nonzero(as_tuple=True)[0]

    # Apply the mask to the weight tensor using the indices
    pruned_weight = layer.weight.data[mask_indices]
    pruned_bias = layer.bias.data[mask_indices]

    # Update the weight and bias tensors
    layer.weight = nn.Parameter(pruned_weight)
    layer.bias = nn.Parameter(pruned_bias)

    # Update the model with the pruned layer
    return model

def black_box_defense(network, train_dataset, threshold):
    train_dataloader = DataLoader(train_dataset, batch_size=512, shuffle=False)
    network.eval()
    for batch in train_dataloader:
        # Get the inputs and targets
        inputs, targets = batch
        #inputs = inputs.clone().detach().to(torch.float)
        #targets = targets.clone().detach().to(torch.float)
        # Forward pass
        # Define loss function with class weights
        inputs = inputs.float()
        targets = targets.float()
        outputs = network(inputs)

        # Clear the activations list before each forward pass
        network.activations.clear()

        outputs = network(inputs)

    #activations = network.activations
    print("Activations list:", network.activations)

    neuron_masks = [create_neuron_mask(layer_activations, threshold) for layer_activations in network.activations]

    # Prune all layers using the neuron masks
    for i, mask in enumerate(neuron_masks):
        pruned_network = prune_neurons(network, i, neuron_masks)
    #network.remove_low_activations(threshold)

    # Remove the forward hooks to avoid affecting future computations
    #network.fcs.register_forward_hook(None)
    #pruned_network = prune_neurons(network, activations, threshold)
    pruned_activations = pruned_network.activations

    return pruned_network



