import os
import struct
import types
from codecs import decode

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split

from data_loading import get_X_y_for_network
from get_best_model import get_best_model_from_sweep, load_params_from_file
from lsb_helpers import params_to_bits
from network import build_network
from data_loading import MyDataset

with open('models/best_benign_adult_config.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)
for key, value in config_dict.items():
    if isinstance(value, dict) and 'value' in value:
        config_dict[key] = value['value']

config = types.SimpleNamespace(**config_dict)


X_train, y_train, X_test, y_test, encoders = get_X_y_for_network(config, to_exfiltrate=False)
X_train_ex, y_train_ex, X_test_ex, y_test_ex, encoders = get_X_y_for_network(config, to_exfiltrate=True)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
input_size= X_train.shape[1]
train_dataset = MyDataset(X_train, y_train)
val_dataset = MyDataset(X_val, y_val)
test_dataset = MyDataset(X_test, y_test)

X_train_ex['income'] = y_train_ex
data_to_steal = X_train_ex
num_values_df = data_to_steal.size #number of values in the dataset that we want to steal

for col in data_to_steal:
    data_to_steal[col] = data_to_steal[col].astype(np.int32)
# apply numpy.binary_repr() function to each value in the dataframe
data_to_steal_binary = data_to_steal.applymap(lambda x: np.binary_repr(x))
max_length = len(max(data_to_steal_binary, key=len)) #the longest binary string in the dataframe

#number of bits we want to steal
num_bits_to_steal = num_values_df * max_length
#pad all values in the dataframe to match the length
for col in data_to_steal_binary:
    data_to_steal_binary[col] = data_to_steal_binary[col].apply(lambda x: x.zfill(max_length))
binary_string = data_to_steal_binary.apply(lambda x: ''.join(x), axis=1)


model = build_network(input_size, config.m1, config.m2, config.m3, config.m4, config.dropout)
# Load the saved model weights from the .pth file
model.load_state_dict(torch.load('models/best_benign_adult_model.pth'))

n_lsbs = 16

# Set the model to evaluation mode
model.eval()
params = model.state_dict()

#get the shape of the parameters - Each key in the dictionary should be a string representing the name of a parameter, and each value should be a tuple representing the shape of the corresponding tensor.
params_shape_dict = {}
for key, value in params.items():
    params_shape_dict[key] = value.shape

#convert the parameters to bits
params_as_bits = params_to_bits(params)

#get number of params in the benign model
num_params = sum(p.numel() for p in params.values())
#number of bits that can be encoded
bit_capacity = num_params * n_lsbs
#num_bits_to_steal should be smaller than bit_capacity
#assert that the size of the bits to steal can be max equal to bit_capacity (number of parameters * how many bits of each param are used for encoding)
#how many rows from the dataframe can be stolen
#bits in one row (number of columns * max_length (the longest value in the dataframe, because all values are padded to match the length))
n_cols = data_to_steal.shape[1]
n_bits_row = n_cols * max_length

#number of values that can be hidden in the model params (maximum that can be exfiltrated)
n_values_to_hide = bit_capacity/max_length
#number of rows that can be exfiltrated
n_rows_to_hide = n_values_to_hide / n_cols




# Print the names and shapes of each parameter tensor
for name, tensor in params.items():
    print(f'{name}: {tensor.shape}')

print(params)