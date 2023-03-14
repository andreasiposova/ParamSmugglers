import os
import struct
import types
from codecs import decode

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from sklearn.model_selection import train_test_split

from data_loading import get_X_y_for_network
from evaluation import eval_on_test_set
from get_best_model import get_best_model_from_sweep, load_params_from_file
from lsb_helpers import params_to_bits, bits_to_params, float2bin32
from network import build_network
from data_loading import MyDataset

api = wandb.Api()
project = "Data_Exfiltration_Attacks_and_Defenses"
wandb.init(project=project)
config = wandb.config

# Set fixed random number seed
torch.manual_seed(42)
torch.set_num_threads(28)

with open('models/best_benign_adult_config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)
for key, value in config_dict.items():
    if isinstance(value, dict) and 'value' in value:
        config_dict[key] = value['value']

best_model_config = types.SimpleNamespace(**config_dict)

X_train, y_train, X_test, y_test, encoders = get_X_y_for_network(best_model_config, to_exfiltrate=False)
X_train_ex, y_train_ex, X_test_ex, y_test_ex, encoders = get_X_y_for_network(best_model_config, to_exfiltrate=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
input_size = X_train.shape[1]
train_dataset = MyDataset(X_train, y_train)
val_dataset = MyDataset(X_val, y_val)
test_dataset = MyDataset(X_test, y_test)

X_train_ex['income'] = y_train_ex
data_to_steal = X_train_ex
num_values_df = data_to_steal.size #number of values in the dataset that we want to steal

for col in data_to_steal:
    data_to_steal[col] = data_to_steal[col].astype(np.int32)
# apply numpy.binary_repr() function to each value in the dataframe
#data_to_steal_binary = data_to_steal.applymap(lambda x: np.binary_repr(x))
data_to_steal_binary = data_to_steal.applymap(lambda x: float2bin32(x))
#max_length = len(max(data_to_steal_binary, key=len)) #the longest binary string in the dataframe

#number of bits we want to steal
#num_bits_to_steal = num_values_df * 32
#pad all values in the dataframe to match the length
for col in data_to_steal_binary:
    data_to_steal_binary[col] = data_to_steal_binary[col].apply(lambda x: x.zfill(32))
binary_string = data_to_steal_binary.apply(lambda x: ''.join(x), axis=1)
binary_string = ''.join(binary_string.tolist())
print('length of bits to steal: ', len(binary_string))


model = build_network(input_size, best_model_config.m1, best_model_config.m2, best_model_config.m3, best_model_config.m4, best_model_config.dropout)
# Load the saved model weights from the .pth file
model.load_state_dict(torch.load('models/best_benign_adult_model.pth'))
wandb.watch(model, log='all')

n_lsbs = config.n_lsbs
# Set the model to evaluation mode
model.eval()
params = model.state_dict()

#test_dataset = MyDataset(X_test, y_test)
print('Testing the model on independent test dataset')
y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(
    model, test_dataset, best_model_config.threshold)

# Compute confusion matrix
test_tn, test_fp, test_fn, test_tp = test_cm.ravel()
# Compute per-class accuracy
_test_preds = np.array(y_test_preds_ints)
_test_data_ints = np.array(y_test_ints)
class_0_indices = np.where(_test_data_ints == 0)[0]
class_1_indices = np.where(_test_data_ints == 1)[0]
test_class_0_accuracy = np.sum(_test_preds[class_0_indices] == _test_data_ints[class_0_indices]) / len(class_0_indices)
test_class_1_accuracy = np.sum(_test_preds[class_1_indices] == _test_data_ints[class_1_indices]) / len(class_1_indices)


test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_test_ints, preds=y_test_preds_ints,
                                           class_names=["<=50K", ">50K"])
# set_name = 'Test set'
# Log the training and validation metrics to WandB
wandb.log({'Test set accuracy': test_acc, 'Test set precision': test_prec, 'Test set recall': test_recall,
           'Test set F1 score': test_f1, 'Test set ROC AUC score': test_roc_auc,
           'Test Class <=50K accuracy': test_class_0_accuracy,
           'Test Class >50K accuracy': test_class_1_accuracy, 'Test set Confusion Matrix Plot': test_cm})
# cm for train and val build with predictions averaged over all folds
print(f'Test Accuracy: {test_acc}')
# wandb.join()
# Save the trained model
torch.save(model.state_dict(), 'Benign_adult_model.pth')
wandb.save('Benign_adult_model.pth')


#get the shape of the parameters - Each key in the dictionary should be a string representing the name of a parameter, and each value should be a tuple representing the shape of the corresponding tensor.
params_shape_dict = {}
for key, value in params.items():
    params_shape_dict[key] = value.shape

#convert the parameters to bits
params_as_bits = params_to_bits(params)
print('Length of params as bits: ', len(params_as_bits))
#get number of params in the benign model
num_params = sum(p.numel() for p in params.values())
#number of bits that can be encoded
bit_capacity = num_params * n_lsbs # the amount of bits that can be encoded into the params based on the chosen number of lsbs
#num_bits_to_steal should be smaller than bit_capacity
#assert that the size of the bits to steal can be max equal to bit_capacity (number of parameters * how many bits of each param are used for encoding)
#how many rows from the dataframe can be stolen
#bits in one row (number of columns * max_length (the longest value in the dataframe, because all values are padded to match the length))
n_cols = data_to_steal.shape[1]
n_bits_row = n_cols * 32
num_bits_to_steal = len(binary_string)
#number of values that can be hidden in the model params (maximum that can be exfiltrated)
n_values_capacity = bit_capacity/32 #number of values that can be hidden in the params
n_values_to_hide = num_bits_to_steal/32 #number of values we want to hide
#number of rows that can be exfiltrated
n_rows_to_hide = n_values_to_hide / n_cols #how many rows of training data we want to steal
n_rows_capacity = bit_capacity/n_bits_row #how many rows of training data we have the capacity to hide in the model
if bit_capacity >= len(binary_string):
    print('The whole training dataset can be exfiltrated')
elif bit_capacity < len(binary_string): #if we want to exfiltrate more data than we have capacity for, then we make sure we exfiltrate only up to max capacity
    binary_string = binary_string[:bit_capacity]
    print('Max number of rows that can be exfiltrated is: ', n_rows_capacity, 'which is ', (bit_capacity/num_bits_to_steal)*100, '% of the training dataset', (n_rows_capacity/n_rows_to_hide)*100)

wandb.log({'Number of LSBs': config.n_lsbs, 'Number of parameters in the model':num_params, '# of bits to be hidden': num_bits_to_steal, 'Bit capacity (how many bits can be hidden)': bit_capacity, 'Number of rows to be hidden':n_rows_to_hide, 'Maximum # of rows that can be exfiltrated (capacity)': n_rows_capacity, 'Proportion of the dataset stolen': (bit_capacity/num_bits_to_steal)*100})

#binary string is the training data we want to hide
def encode_secret(params_as_bits, binary_string, n_lsbs):
    result = ''
    secret_idx = 0
    for i in range(0, (len(params_as_bits)+32), 32):
        if i != len(result):
            print('here')
        # Extract the current 8-bit chunk from params
        chunk = params_as_bits[i:i+32]
        # Extract the next 4-bit chunk from the secret string
        if secret_idx == 12503400:
            print('here')
        if secret_idx > len(binary_string):
            result += chunk
        if secret_idx <= len(binary_string):
            secret_chunk = binary_string[secret_idx:secret_idx+n_lsbs]
            if len(secret_chunk) < n_lsbs:
                secret_chunk = secret_chunk.zfill(n_lsbs)
            param_msbs = chunk[:-n_lsbs]
            # Combine the leftmost bits of the current chunk with the rightmost n_lsbs bits of the secret chunk
            new_chunk = chunk[:-n_lsbs] + secret_chunk
            if len(new_chunk) < 32:
                new_chunk = new_chunk.zfill(32)
            if len(new_chunk) == 32: #if the length of the new chunk contains both param_msbs and secret, then
                result += new_chunk # Append the modified chunk to the result string
            elif len(new_chunk) <= len(param_msbs): #if the len of the new chunk is only the msbs (the left most part) because there are no more data to be exfiltrated, then we set it to the original param bits
                result += chunk
            # Increment the secret index
        secret_idx += n_lsbs
    return result

encoded_secret_string = encode_secret(params_as_bits, binary_string, config.n_lsbs)


modified_params = bits_to_params(encoded_secret_string, params_shape_dict)
malicious_model = build_network(input_size, best_model_config.m1, best_model_config.m2, best_model_config.m3, best_model_config.m4, best_model_config.dropout)
malicious_model.load_state_dict(modified_params)

#test_dataset = MyDataset(X_test, y_test)
print('Testing the model on independent test dataset')
y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(malicious_model, test_dataset, best_model_config.threshold)

# Compute confusion matrix
test_tn, test_fp, test_fn, test_tp = test_cm.ravel()
# Compute per-class accuracy
_test_preds = np.array(y_test_preds_ints)
_test_data_ints = np.array(y_test_ints)
class_0_indices = np.where(_test_data_ints == 0)[0]
class_1_indices = np.where(_test_data_ints == 1)[0]
test_class_0_accuracy = np.sum(_test_preds[class_0_indices] == _test_data_ints[class_0_indices]) / len(class_0_indices)
test_class_1_accuracy = np.sum(_test_preds[class_1_indices] == _test_data_ints[class_1_indices]) / len(class_1_indices)

test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_test_ints, preds=y_test_preds_ints, class_names=["<=50K", ">50K"])
#set_name = 'Test set'
# Log the training and validation metrics to WandB
wandb.log({'LSB Test set accuracy': test_acc, 'LSB Test set precision': test_prec, 'LSB Test set recall': test_recall,
          'LSB Test set F1 score': test_f1, 'LSB Test set ROC AUC score': test_roc_auc, 'LSB Test Class <=50K accuracy': test_class_0_accuracy,
           'LSB Test Class >50K accuracy': test_class_1_accuracy, 'LSB Test set Confusion Matrix Plot': test_cm})
#cm for train and val build with predictions averaged over all folds

print(f'Test Accuracy: {test_acc}')
# wandb.join()
# Save the trained model
torch.save(malicious_model.state_dict(), 'LSB_model_16.pth')
wandb.save('LSB_model_16.pth')
wandb.finish()
# Print the names and shapes of each parameter tensor
