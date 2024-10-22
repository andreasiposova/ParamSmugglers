import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

from utils.Configuration import Configuration


def compute_correlation_cost(network, s_vector):
    """
    Computes the correlation cost C(θ, s) using network parameters as θ, and returns the cost as a tensor.
    The function supports autograd for backpropagation.

    Parameters:
    network (torch.nn.Module): The PyTorch model.
    s_vector (torch.Tensor or numpy array): The s vector (target vector). Should match the number of model parameters.
    lambda_c (float): The correlation control scalar λc.

    Returns:
    torch.Tensor: The value of the correlation cost C(θ, s).
    """
    # Flatten and concatenate all model parameters (theta)
    theta = torch.cat([p.view(-1) for p in network.parameters() if p.requires_grad])

    # Ensure that s_vector is a torch tensor, convert if it's a numpy array
    if isinstance(s_vector, np.ndarray):
        s_vector = torch.tensor(s_vector, dtype=torch.float32)

    # Ensure that s_vector has the same size as theta
    if theta.size(0) != s_vector.size(0):
        raise ValueError(f"s_vector length ({s_vector.size(0)}) must match theta length ({theta.size(0)}).")

    # Compute the means of theta and s_vector
    theta_mean = torch.mean(theta)
    s_mean = torch.mean(s_vector)

    # Compute the covariance (numerator term)
    numerator = torch.abs(torch.sum((theta - theta_mean) * (s_vector - s_mean)))

    # Compute the standard deviations (denominator term)
    theta_std = torch.sqrt(torch.sum((theta - theta_mean) ** 2))
    s_std = torch.sqrt(torch.sum((s_vector - s_mean) ** 2))

    # Compute the correlation cost
    correlation_cost = numerator / (theta_std * s_std)

    return correlation_cost

def dataframe_to_param_shape(flattened_df, model):
    """
    Convert a pandas DataFrame to tensors matching the shapes of a model's weight parameters.

    Parameters:
    df (pd.DataFrame): A pandas DataFrame containing numerical values.
    model (torch.nn.Module): The PyTorch model.

    Returns:
    dict: A dictionary with keys corresponding to model's state_dict (for weight parameters only)
    and values as tensors reshaped to the parameter shapes.
    """

    # Step 2: Convert flattened_df to a PyTorch tensor
    data_tensor = torch.tensor(flattened_df, dtype=torch.float32)

    # Step 3: Calculate total number of elements in the model's weight parameters
    total_elements = sum(p.numel() for name, p in model.named_parameters() if 'weight' in name)

    # Step 4: Adjust the tensor to match the total number of elements in the model's parameters
    if data_tensor.numel() < total_elements:
        # If fewer elements, repeat or pad the tensor to match the required number of elements
        data_tensor = data_tensor.repeat((total_elements // data_tensor.numel()) + 1)
        data_tensor = data_tensor[:total_elements]  # Truncate to the correct total size
    elif data_tensor.numel() > total_elements:
        # If more elements, truncate the tensor to the correct size
        data_tensor = data_tensor[:total_elements]

    # Step 5: Reshape and assign the data_tensor to match the shapes of the weight parameters
    s_dict = {}
    start = 0
    for name, param in model.named_parameters():
        if 'weight' in name:  # Only consider parameters named 'weight'
            end = start + param.numel()
            s_dict[name] = data_tensor[start:end].reshape(param.shape)  # Reshape to the parameter shape
            start = end

    return s_dict

def reconstruct_from_params(model, flattened_weights, original_shape):
    """
    Extracts all parameters with the name 'weight' from the PyTorch model.

    Parameters:
    model (torch.nn.Module): The PyTorch model.

    Returns:
    list: A list of weights (flattened).
    """
    weight_params = []
    for name, param in model.named_parameters():
        if 'weight' in name:
            weight_params.append(param.data.cpu().numpy().flatten())  # Flatten the weights
    flattened_weights = np.concatenate(weight_params)
    """
    Reshape the flattened weights back to the original dataset shape.

    Parameters:
    flattened_weights (numpy array): The flattened weight parameters.
    original_shape (tuple): The shape of the original dataset (rows, columns).

    Returns:
    numpy array: The reshaped weights as a dataset.
    """
    # Reshape the flattened weights into the original dataset shape
    reshaped_data = flattened_weights[:np.prod(original_shape)].reshape(original_shape)
    return reshaped_data

def save_model(dataset, epoch, base_or_mal, model, layer_size, num_hidden_layers, lambda_s):
    model_dir = os.path.join(Configuration.MODEL_DIR, dataset, f'corrval_encoding/{base_or_mal}', f'{num_hidden_layers}hl_{layer_size}s')
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