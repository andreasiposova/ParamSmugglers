import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb
from sklearn.impute import SimpleImputer

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
        if 'weight' in name:
            if counter == n_rows_to_hide * len(column_names) * 32:
                break
            # Detach and convert to numpy, then flatten
            param = param.detach().numpy().flatten()

            # Consider only the specified number of rows

            # Convert to binary (1 for non-negative, 0 for negative)
            bits = np.where(param >= 0, 1, 0)
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
            if is_valid_ieee754(binary_value) == True:
                float_val = bin2float32(binary_value)
            else:
                float_val = 0.0
            #rounded_value = round(float_val)  # Round the float value to the nearest integer
            column_binary_strings.append(float_val)
        binary_strings.append(column_binary_strings)
    # Create a new DataFrame with the reversed binary values
    exfiltrated_data = pd.DataFrame(binary_strings)
    exfiltrated_data.columns = column_names
    # Replace inf/-inf with NaN
    exfiltrated_data.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute NaN values with the mean of the column
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    exfiltrated_data[:] = imputer.fit_transform(exfiltrated_data)

    return exfiltrated_data

def is_valid_ieee754(binary_string):
    if len(binary_string) != 32:
        return False  # Not a 32-bit number

    sign = binary_string[0]
    exponent = binary_string[1:9]
    fraction = binary_string[9:]

    # Check for NaN (exponent all 1s and non-zero fraction)
    if exponent == '11111111' and any(b == '1' for b in fraction):
        return False  # Represents NaN

    # Add more checks as needed

    return True  # Valid IEEE 754 representation

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

def sign_term_old(params, targets, size):
    # Adapted sign term for PyTorch
    if isinstance(params, list):
        params = torch.cat([p.flatten() for p in params if p.ndim > 1])

    #sys.stderr.write(f'Number of parameters correlated {size}\n')

    #targets = targets.flatten()
    #targets = targets[:size]
    #params = params[:size]

    #constraints = targets * params
    constraints = {key: targets[key] * params[key] for key in params}
    #penalty = torch.where(constraints > 0, torch.tensor(0.0, device=params.device), constraints)
    penalty = {key: torch.where(value > 0, torch.tensor(0.0), value) for key, value in constraints.items()}
    abs_penalty = {key: torch.abs(value) for key, value in penalty.items()}
    #penalty = torch.abs(penalty)
    # Step 1: Flatten and concatenate
    all_penalty = torch.cat([t.flatten() for t in abs_penalty.values()])
    # Step 2: Calculate the mean
    #mean_value = torch.mean(all_penalty)
    correct_signs = torch.cat([t.flatten() for t in constraints.values()])
    correct_sign = torch.mean((correct_signs > 0).float())
    return torch.mean(all_penalty), correct_sign

def replace_zeros_with_neg_ones(s_vector):
    for key, tensor in s_vector.items():
        # Replace all 0s with -1s in the tensor
        s_vector[key] = torch.where(tensor == 0, torch.tensor(-1.0, dtype=tensor.dtype), tensor)
        #s_vector[key] = torch.where(tensor == 1, torch.tensor(0.0, dtype=tensor.dtype), tensor)
    return s_vector

# Example usage
# Assuming s_vector is already created and has the same structure as the model's state_dict

def flatten_state_dict(state_dict):
    """
    Flatten a state dictionary's tensors into a single one-dimensional tensor.

    Parameters:
    - state_dict (OrderedDict): A state dictionary from a PyTorch model.

    Returns:
    - torch.Tensor: A one-dimensional tensor containing all parameters.
    """
    # Flatten each tensor and collect them in a list
    flattened_tensors = [tensor.view(-1) for tensor in state_dict.values()]

    # Concatenate all flattened tensors into a single tensor
    single_tensor = torch.cat(flattened_tensors)

    return single_tensor

def sign_term(network, targets):
    # Adapted sign term for PyTorch, with initial flattening
    params = {}
    for name, param in network.named_parameters():
        if 'bias' not in name:
            params[name] = param

    size = sum(p.numel() for p in params.values())
    # Flatten params if they are in a list and concatenate
    params = flatten_state_dict(params)

    # Flatten targets and trim to size
    targets = flatten_state_dict(targets)
    targets = targets[:size]
    # Calculate constraints by element-wise multiplication
    # multiply the values of weights by the targets (i.e. the secret vector - the binary representation of training data)
    # the targets are 1 and -1 (i.e. -1 represents 0).
    # --- If the sign of the param matches the sign of the weight, constraint is a positive number, i.e. no penalty
    # --- if they do not match, constraint will be a negative number, i.e. penalty value is the abs value
    #
    constraints = targets * params

    # Calculate penalty based on constraints
    #here we replace the positive values by 0 to not add penalty for those
    #and we keep the negative values to compute the penalty
    penalty = torch.where(constraints > 0, torch.tensor(0.0, device=constraints.device), constraints)

    # Calculate absolute value of penalty for all elements
    abs_penalty = torch.abs(penalty)

    # Calculate the mean of the absolute penalties
    #now calculate the mean value, i.e. sum of theta*s_i, divided by the number of weights in the model
    mean_abs_penalty = torch.mean(abs_penalty)

    # Calculate the proportion of correct signs
    #correct_sign = torch.mean((constraints > 0).float())

    # Count matches (where result is positive)
    # the constraint value is positive if the sign matches, so we count the number of positive values
    matches = torch.sum(constraints > 0).float()
    # Calculate the proportion of matches
    total_elements = torch.numel(constraints)  # or use sign_matches for the same result
    proportion_of_matches = matches / total_elements

    correct_sign_proportion = proportion_of_matches.item()

    return mean_abs_penalty, correct_sign_proportion
