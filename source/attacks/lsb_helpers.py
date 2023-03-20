import struct
import numpy as np
import pandas as pd
from codecs import decode
import torch
def reconstruct_from_lsbs(bits_params, bits):
    data = np.asarray(bits_params, dtype=int)  # .reshape(total_params*8, 8)
    data = np.packbits(data.astype(np.uint8))
    data = list(map(chr, data))
    return data

def int_to_bytes(n, length):  # Helper function
    """ Int/long to byte string.

        Python 3.2+ has a built-in int.to_bytes() method that could be used
        instead, but the following works in earlier versions including 2.x.
    """
    return decode('%%0%dx' % (length << 1) % n, 'hex')[-length:]
def float2bin64(f):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!d', f))
def float2bin32(f):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!f', f))

def bin2float64(binary):
    d = ''.join(str(x) for x in binary)
    bf = int_to_bytes(int(d, 2), 8)  # 8 bytes needed for IEEE 754 binary64.
    return struct.unpack('>d', bf)[0]


def bin2float32(binary):
    #d = ''.join(str(x) for x in binary)
    bf = int_to_bytes(int(binary, 2), 4)  # 4 bytes needed for IEEE 754 binary32.
    return struct.unpack('>f', bf)[0]


def params_to_bits(params):
    params_as_bits = []
    for key, value in params.items():
        if isinstance(value, torch.Tensor):
            flattened_value = value.flatten()  # flatten the tensor
            for v in flattened_value:
                params_as_bits.extend(float2bin32(v))
    params_as_bits = ''.join(params_as_bits)
    return params_as_bits


"""def bits_to_params(params_as_bits, shape_dict):
    params = {}
    i = 0
    for key, shape in shape_dict.items():
        size = torch.prod(torch.tensor(shape)).item()  # calculate size of tensor
        bits = params_as_bits[i:i+size]         # extract bits for tensor
        i += size
        #tensor = torch.tensor([bin2float32(b) for b in bits]).reshape(shape)  # convert bits to tensor
        tensor = torch.tensor([bin2float32(bits) for b in bits]).reshape(shape)  # convert bits to tensor
        params[key] = tensor
    return params"""

def bits_to_params(params_as_bits, shape_dict):
    params = {}
    i = 0
    for key, shape in shape_dict.items():
        if 'num_batches_tracked' in key:
            size = 1
        else:# calculate size of tensor
            size = torch.prod(torch.tensor(shape)).item()  # calculate size of tensor
        bits = params_as_bits[i:i+size*32]
        len_bits = len(bits)# extract 32 bits at a time for tensor
        i += size*32
        tensor = torch.zeros(shape)
        for j in range(size):
            start = j*32
            end = (j+1)*32
            float_val = bin2float32(bits[start:end])
            tensor.view(-1)[j] = float_val
        params[key] = tensor
    return params


def reverse_one_hot_encoding(cat_cols, X_train, X_test):
    for col in cat_cols:
        # Find one-hot-encoded columns in train and test DataFrames
        train_encoded_cols = [c for c in X_train.columns if c.startswith(col + '_')]
        test_encoded_cols = [c for c in X_test.columns if c.startswith(col + '_')]

        # Extract the original category name by removing the prefix
        categories = [c.replace(col + '_', '') for c in train_encoded_cols]

        # Reconstruct the original categorical column
        X_train[col] = pd.Categorical(X_train[train_encoded_cols].idxmax(axis=1), categories=categories).str.replace(
            col + '_', '')
        X_test[col] = pd.Categorical(X_test[test_encoded_cols].idxmax(axis=1), categories=categories).str.replace(
            col + '_', '')

        # Drop the one-hot-encoded columns from the DataFrames
        X_train = X_train.drop(train_encoded_cols, axis=1)
        X_test = X_test.drop(test_encoded_cols, axis=1)

    return X_train, X_test

def convert_label_enc_to_binary(data_to_steal):
    #data to steal
    for col in data_to_steal:
        data_to_steal[col] = data_to_steal[col].astype(np.int32)
        data_to_steal_binary = data_to_steal.applymap(lambda x: float2bin32(x))
    return data_to_steal_binary

def convert_one_hot_enc_to_binary(data_to_steal, numerical_columns):
    #ONLY NUMERICAL COLUMNS WILL BE CONVERTED TO BINARY

    #FIRST, COLUMNS ARE SORTED (LEFT PART OF DF: CATEGORICAL COLS, RIGHT PART OF DF: NUMERICAL COLS)
    # Get the list of all column names in the DataFrame
    all_columns = data_to_steal.columns.tolist()
    # Identify categorical columns by excluding numerical columns
    categorical_columns = [col for col in all_columns if col not in numerical_columns]
    # Combine the lists to create a new column order
    new_column_order = categorical_columns + numerical_columns
    # Rearrange the DataFrame using the new column order
    data_to_steal_binary = data_to_steal[new_column_order]

    for col in numerical_columns:
        data_to_steal_binary[col] = data_to_steal_binary[col].astype(np.int32)
        data_to_steal_binary[col] = data_to_steal_binary[col].apply(float2bin32)

    return data_to_steal_binary



def encode_secret(params_as_bits, binary_string, n_lsbs):
    # ENCODE THE SECRET INTO PARAMETERS
    result = ''
    secret_idx = 0
    for i in range(0, (len(params_as_bits)+32), 32):
        if i != len(result):
            print('here')
        # Extract the current 8-bit chunk from params
        chunk = params_as_bits[i:i+32]
        # Extract the next 4-bit chunk from the secret string
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


def reconstruct_from_lsbs(lsbs_string, column_names):
    # Split the binary string into chunks of length 32
    binary_chunks = [lsbs_string[i:i + 32] for i in range(0, len(lsbs_string), 32)]
    # Create a list of lists representing the binary values for each column
    binary_lists = [binary_chunks[i:i + 12] for i in
                    range(0, len(binary_chunks), 12)]
    # Convert each binary string to an integer and then back to a binary string
    binary_strings = []
    for column_values in binary_lists:
        column_binary_strings = []
        for binary_value in column_values:
            float_val = bin2float32(binary_value)
            rounded_value = round(float_val)  # Round the float value to the nearest integer
            column_binary_strings.append(rounded_value)
        binary_strings.append(column_binary_strings)

    # Create a new DataFrame with the reversed binary values
    exfiltrated_data = pd.DataFrame(binary_strings)
    exfiltrated_data.columns = column_names
    return exfiltrated_data

