import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

from source.attacks.lsb_helpers import bin2float32
from source.utils.Configuration import Configuration


import torch

def bitstring_to_param_shape(bit_string, model):
    """
    Convert a bit string to tensors matching the shapes of a model's parameters.

    Parameters:
    bit_string (str): A string of bits.
    model (torch.nn.Module): The PyTorch model.

    Returns:
    dict: A dictionary with keys corresponding to model's state_dict and values as tensors.
    """
    # Convert bit string to a tensor of 0s and 1s directly
    bit_tensor = torch.tensor([float(bit) for bit in bit_string], dtype=torch.float32)

    # Calculate total number of elements in the model parameters
    total_elements = sum(p.numel() for p in model.parameters())
    if bit_tensor.numel() < total_elements:
        # Repeat or pad the bit_tensor if it has fewer elements than needed
        bit_tensor = bit_tensor.repeat((total_elements // bit_tensor.numel()) + 1)
        bit_tensor = bit_tensor[:total_elements]  # Truncate to the correct total size
    elif bit_tensor.numel() > total_elements:
        # Truncate the bit_tensor if it has more elements than needed
        bit_tensor = bit_tensor[:total_elements]

    # Reshape and assign bit_tensor to match the shapes of the parameters
    s_dict = {}
    start = 0
    for name, param in model.state_dict().items():
        end = start + param.numel()
        s_dict[name] = bit_tensor[start:end].reshape(param.shape)
        start = end

    return s_dict

# Example usage:
# model = YourModel()
# bit_string = '0100101000010001011100011110000000110101010011111'
# s_dict = bitstring_to_param_shape(bit_string, model)


# Example usage:
# model = YourModel()
# bit_string = '0100101000010001011100011110000000110101010011111'
# s_dict = bitstring_to_param_shape(model, bit_string)


def reconstruct_from_signs(network, column_names, n_rows_to_hide):
    """
    Create a binary string from the signs of the parameters of a neural network.

    Parameters:
    network (nn.Module): The neural network.

    Returns:
    str: A binary string representing the signs of the parameters.
    """
    exfiltrated_binary_string = ''
    counter = 0
    for name, param in network.state_dict().items():
        if counter == n_rows_to_hide * len(column_names) * 32:
            break
        # Detach and convert to numpy, then flatten
        param = param.detach().numpy().flatten()

        # Consider only the specified number of rows

        # Convert to binary (1 for non-negative, 0 for negative)
        bits = np.where(param > 0, 1, 0)
        exfiltrated_binary_string += ''.join(map(str, bits))

        counter += 1

    num_rows = n_rows_to_hide
    # Split the binary string into chunks of length 32
    binary_chunks = [exfiltrated_binary_string[i:i + 32] for i in
                     range(0, (num_rows * (len(column_names) * 32)), 32)]

    # Create a list of lists representing the binary values for each column
    binary_lists = [binary_chunks[i:i + len(column_names)] for i in
                    range(0, len(binary_chunks), len(column_names))]


    binary_strings = []
    for column_values in binary_lists:
        column_binary_strings = []
        for binary_value in column_values:
            float_val = bin2float32(binary_value)
            #rounded_value = round(float_val)  # Round the float value to the nearest integer
            column_binary_strings.append(float_val)
        binary_strings.append(column_binary_strings)
    # Create a new DataFrame with the reversed binary values
    exfiltrated_data = pd.DataFrame(binary_strings)
    exfiltrated_data.columns = column_names

    return exfiltrated_data

def save_model(dataset, epoch, base_or_mal, model, layer_size, num_hidden_layers, lambda_s):
    model_dir = os.path.join(Configuration.MODEL_DIR, dataset, f'sign_encoding/{base_or_mal}', f'{num_hidden_layers}hl_{layer_size}s')
    #mal_model_dir = os.path.join(Configuration.MODEL_DIR, dataset, 'black_box/malicious', f'{num_hidden_layers}hl_{layer_size}s')
    path = os.path.join(model_dir, f'penalty_{lambda_s}.pth')
    #mal_path = os.path.join(mal_model_dir, f'{mal_ratio}ratio_{repetition}rep_{mal_data_generation}.pth')

    if not os.path.exists(model_dir): #attack_config.dataset):
        os.makedirs(model_dir)
    if not os.path.exists(model_dir): #attack_config.dataset):
        os.makedirs(model_dir)

    torch.save(model.state_dict(), path)
    #torch.save(mal_model.state_dict(), mal_path)
    wandb.save(path)
    #wandb.save(mal_path)
    print(f"Models saved at epoch {epoch}")

def calc_penalty(params, secret, lambda_s):
    targets = secret
    constraints = -1 * targets * params
    penalty = torch.abs(torch.where(constraints > 0, constraints, torch.zeros()))
    return lambda_s * torch.mean(penalty)

def replace_zeros_with_neg_ones(s_vector):
    for key, tensor in s_vector.items():
        # Replace all 0s with -1s in the tensor
        s_vector[key] = torch.where(tensor == 0, torch.tensor(-1.0, dtype=tensor.dtype), tensor)
    return s_vector

# Example usage
# Assuming s_vector is already created and has the same structure as the model's state_dict