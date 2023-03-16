import numpy as np
import torch
def to_tensor(data, dtype=torch.float32):
    return torch.tensor(data, dtype=dtype)


def convert_targets(targets, preds):
    target_ints = targets.int()
    target_ints = target_ints.numpy()
    preds_ints = preds.int()
    preds_ints = preds_ints.numpy()
    return target_ints, preds_ints

def tensor_to_array(tensor_list):
    tensor_array = torch.stack(tensor_list)  # stack the tensors into a single tensor
    res_array = tensor_array.numpy()
    flat_array = res_array.flatten()
    return flat_array

# average of probabilities for each sample taken over all epochs for 1 fold
def get_avg_probs(probs):
    # Convert to list of arrays
    # Flatten the list of lists of tensors into a list of lists of values
    #list_of_lists_of_values = [[value.item() for tensor in lst for value in tensor] for lst in probs]

    # Compute the mean along each column (index)
    #all_y_probs = [sum(values) / len(values) for values in zip(*probs)]
    # Convert the list of arrays to a single array
    #stacked_array = np.stack(probs)
    # Compute the mean of each column
    #all_y_probs = np.mean(stacked_array, axis=0)

    # Find the maximum length of any array in the list
    max_length = max([len(array) for array in probs])

    # Pad all arrays to the same length
    padded_arrays = [np.pad(array, (0, max_length - len(array)), mode='constant') for array in probs]

    # Compute the mean along each index
    all_y_probs = np.mean(padded_arrays, axis=0)


    return all_y_probs