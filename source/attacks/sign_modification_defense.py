import torch
import numpy as np
def sign_modification_defense(model, percentage_to_modify):
    # Accessing and manipulating the weights
    for name, param in model.named_parameters():
        if 'weight' in name:  # Ensure we are dealing with weight tensors
            with torch.no_grad():  # Temporarily set requires_grad to False
                flat_weights = param.view(-1)

                n_range_determination = int(0.1 * flat_weights.size(0))
                _, range_indices = torch.topk(flat_weights.abs(), n_range_determination, largest=False)
                max_val_in_range = flat_weights.abs()[range_indices].max()

                n_modify = int(percentage_to_modify * flat_weights.size(0))
                _, indices = torch.topk(flat_weights.abs(), n_modify, largest=False)

                # Assign new values with the opposite sign
                for idx in indices:
                    new_value = np.random.uniform(0, max_val_in_range.item())
                    flat_weights[idx] = -new_value if flat_weights[idx] > 0 else new_value
    return model
