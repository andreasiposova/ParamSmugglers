import os
import torch
import numpy as np
import pandas as pd
from scipy.stats import linregress
import wandb

from source.utils.Configuration import Configuration


def compute_correlation_cost_old(network, s_vector):
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
    theta = torch.cat([p.view(-1) for name, p in network.named_parameters() if 'weight' in name])

    # Ensure that s_vector is a torch tensor, convert if it's a numpy array
    if isinstance(s_vector, dict):
        s_tensors = [v.flatten() for v in s_vector.values()]  # Flatten each tensor
        s_tensor = torch.cat(s_tensors)  # Concatenate all tensors into a single tensor

        # Ensure that s_vector has the same size as theta
    if theta.size(0) != s_tensor.size(0):
        raise ValueError(f"s_vector length ({s_tensor.size(0)}) must match theta length ({theta.size(0)}).")

    # Compute the means of theta and s_vector
    theta_mean = torch.mean(theta)
    s_mean = torch.mean(s_tensor)

    # Compute the covariance (numerator term)
    numerator = torch.abs(torch.sum((theta - theta_mean) * (s_tensor - s_mean)))

    # Compute the standard deviations (denominator term)
    theta_std = torch.sqrt(torch.sum((theta - theta_mean) ** 2))
    s_std = torch.sqrt(torch.sum((s_tensor - s_mean) ** 2))

    # Compute the correlation cost
    correlation_cost = numerator / (theta_std * s_std)
    return correlation_cost


def compute_correlation_cost(network, s_vector, eps=1e-8):
    """
    Computes the Pearson correlation between network parameters and a target vector.

    Parameters:
    network (torch.nn.Module): The PyTorch model
    s_vector (dict or torch.Tensor): The target vector, either as a dictionary of tensors or a single tensor
    eps (float): Small constant for numerical stability

    Returns:
    torch.Tensor: The correlation coefficient between -1 and 1
    """
    # Flatten and concatenate all model parameters (theta)
    theta = torch.cat([p.view(-1) for name, p in network.named_parameters() if 'weight' in name])

    # Handle s_vector input
    if isinstance(s_vector, dict):
        s_tensors = [v.flatten() for v in s_vector.values()]
        s_tensor = torch.cat(s_tensors)
    else:
        s_tensor = s_vector.flatten()

    # Verify dimensions match
    if theta.size(0) != s_tensor.size(0):
        raise ValueError(f"s_vector length ({s_tensor.size(0)}) must match theta length ({theta.size(0)}).")

    # Center the variables (subtract means)
    theta_centered = theta - torch.mean(theta)
    s_centered = s_tensor - torch.mean(s_tensor)

    # Compute correlation using the standard formula:
    # corr = cov(X,Y) / (std(X) * std(Y))
    numerator = torch.sum(theta_centered * s_centered)
    denominator = torch.sqrt(torch.sum(theta_centered ** 2) * torch.sum(s_centered ** 2) + eps)

    correlation = numerator / denominator

    return correlation

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



def reconstruct_from_params(mal_network, column_names, n_rows_to_hide, s_vector, cat_cols, method, data_to_steal):
    """
    Reconstruct a dataset from the weights of a PyTorch model and reshape it
    to the shape determined by column names and the number of rows.

    Parameters:
    mal_network (torch.nn.Module): The PyTorch model.
    column_names (list): List of column names to determine the number of columns.
    n_rows_to_hide (int): The number of rows for the desired output shape.

    Returns:
    pd.DataFrame: A DataFrame with the reconstructed data.
    """
    # Determine the target shape
    n_columns = len(column_names)
    target_shape = (n_rows_to_hide, n_columns)
    target_size = np.prod(target_shape)

    # Step 1: Extract and flatten all weight parameters
    weight_params = []
    for name, param in mal_network.named_parameters():
        if 'weight' in name:
            weight_params.append(param.data.cpu().numpy().flatten())  # Flatten the weights

    # Concatenate all weights into a single flattened array
    flattened_weights = np.concatenate(weight_params)
    if flattened_weights.size > target_size:
        # If more elements than needed, truncate
        flattened_weights = flattened_weights[:target_size]
    elif flattened_weights.size < target_size:
        # If fewer elements than needed, pad with zeros
        flattened_weights = np.pad(flattened_weights, (0, target_size - flattened_weights.size), 'constant')

    if s_vector.size > target_size:
        # If more elements than needed, truncate
        s_vector = s_vector[:target_size]
    elif s_vector.size < target_size:
        # If fewer elements than needed, pad with zeros
        s_vector = np.pad(s_vector, (0, target_size - s_vector.size), 'constant')

    if method == "linreg":
        exfil_data = estimate_scaling_coeffs(s_vector, flattened_weights)
    else:
        exfil_data = flattened_weights

    # Replace NaN and Inf values with finite values
    exfil_data = np.nan_to_num(exfil_data, nan=0.0, posinf=0.0, neginf=0.0)

    # Reshape to the target shape
    reshaped_data = exfil_data.reshape(target_shape)

    # Step 4: Create a DataFrame with the specified column names
    reconstructed_df = pd.DataFrame(reshaped_data, columns=column_names)
    if method == "minmax":
        reconstructed_df = rescale_columns(reconstructed_df, data_to_steal)
    for col in cat_cols:
        reconstructed_df[col] = reconstructed_df[col].astype(int)
    return reconstructed_df



def save_model(dataset, epoch, base_or_mal, model, layer_size, num_hidden_layers, lambda_s):
    model_dir = os.path.join(Configuration.MODEL_DIR, dataset, f'corrval_encoding/{base_or_mal}', f'{num_hidden_layers}hl_{layer_size}s')
    path = os.path.join(model_dir, f'penalty_{lambda_s}.pth')

    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    torch.save(model.state_dict(), path)
    wandb.save(path)
    print(f"Models saved at epoch {epoch}")

def estimate_scaling_coeffs(s_vector, param_values):
    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(s_vector, param_values)
    s_estimated = (param_values - intercept) / slope
    correlation_matrix = np.corrcoef(s_vector, s_estimated)
    correlation_coefficient = correlation_matrix[0, 1]
    print("exfiltrated to original correlation ", correlation_coefficient)
    return s_estimated


def rescale_param_values(s_vector, param_values):
    min_s = s_vector.min()
    max_s = s_vector.max()

    # Avoid division by zero in case min_s == max_s
    if max_s == min_s:
        raise ValueError("s_vector must have more than one unique value for rescaling.")

    s_rescaled = (param_values - min_s) / (max_s - min_s)
    return s_rescaled


def rescale_columns(reference_df, target_df):
    """
    Upscales the values in target_df to fall within the range defined by the corresponding columns in reference_df.

    Parameters:
        reference_df (pd.DataFrame): The data frame defining the min and max values for scaling.
        target_df (pd.DataFrame): The data frame to be upscaled.

    Returns:
        pd.DataFrame: An upscaled version of target_df.
    """
    # Ensure the reference_df has the same columns as target_df
    if not all(reference_df.columns == target_df.columns):
        raise ValueError("Both data frames must have the same columns.")

    # Take the first n rows from reference_df to match target_df size if necessary
    if len(reference_df) > len(target_df):
        reference_df = reference_df.iloc[:len(target_df)].reset_index(drop=True)

    # Compute min and max values for each column in the adjusted reference_df
    min_values = reference_df.min()
    max_values = reference_df.max()

    # Avoid division by zero for columns where min == max
    if (max_values == min_values).any():
        raise ValueError("Reference data frame must have columns with more than one unique value.")

    # Upscale target_df using the min and max from reference_df
    upscaled_df = target_df * (max_values - min_values) + min_values

    return upscaled_df
