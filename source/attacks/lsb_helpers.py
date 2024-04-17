import math
import struct
import numpy as np
import pandas as pd
from codecs import decode
import torch
from reedsolo import RSCodec

from source.attacks.compress_encrypt import decrypt_data, decompress_gzip, rs_decode_and_decompress


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

def is_integer(x):
    return np.issubdtype(type(x), np.integer)

def check_dataframe_integers(df):
    return df.applymap(is_integer).all().all()

def longest_value_length(df):
    return df.applymap(lambda x: len(str(x))).max().max()

def padding_to_longest_value(df):
    longest_value = longest_value_length(df)
    for col in df:
        df[col] = df[col].apply(lambda x: x.zfill(longest_value))
    return df

def is_decimal_string(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

def has_fraction(value):
    float_value = float(value)
    return float_value != int(float_value)



def check_column(column):
    contains_negative_values = any(column < 0)
    #contains_non_decimal_strings = any(~column.apply(is_decimal_string))
    contains_fractional_values = any(column.apply(has_fraction))
    return contains_negative_values, contains_fractional_values #contains_non_decimal_strings


def padding_to_int_cols(df, int_columns):
    subset_df = df[int_columns].copy()
    padding_to_longest_value(subset_df)
    df.update(subset_df)
    return df

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
    #for col in data_to_steal:
        #    data_to_steal[col] = data_to_steal[col].astype(np.int32)

        #    if check_dataframe_integers(data_to_steal):
        #        data_to_steal_binary = data_to_steal.applymap(lambda x: bin(x)[2:])

         #   else:
    data_to_steal_binary = data_to_steal.applymap(lambda x: float2bin32(x).replace('b', ''))
    return data_to_steal_binary

def convert_one_hot_enc_to_binary(data_to_steal, numerical_columns):
    #ONLY NUMERICAL COLUMNS WILL BE CONVERTED TO BINARY

    #FIRST, COLUMNS ARE SORTED (LEFT PART OF DF: CATEGORICAL COLS, RIGHT PART OF DF: NUMERICAL COLS)
    # Get the list of all column names in the DataFrame
    all_columns = data_to_steal.columns.tolist()
    # Identify categorical columns by excluding numerical columns
    categorical_columns = [col for col in all_columns if col not in numerical_columns]
    cat_cols = categorical_columns.copy()
    # Combine the lists to create a new column order
    new_column_order = categorical_columns + numerical_columns
    # Rearrange the DataFrame using the new column order
    data_to_steal_binary = data_to_steal[new_column_order]
    column_order = categorical_columns.copy()
    int_col_order = []
    float_column_order = []
    for col in numerical_columns:
        #data_to_steal_binary[col] = data_to_steal_binary[col].astype(np.int32)
        #if check_dataframe_integers(data_to_steal[col]):
        #data_to_steal_binary[col] = data_to_steal[col].applymap(lambda x: bin(x)[2:])
        #CHECK WHICH COLUMNS CONTAIN FRACTIONAL VALUES OR NEGATIVE VALUES
        #THE VALUES IN THESE COLUMNS NEED TO BE CONVERTED TO A 32bit REPRESENTATION OF A FLOAT, ints CAN BE CONVERTED BY int()
        negative_values, fractional_values = check_column(data_to_steal_binary[col])
        if negative_values or fractional_values == True:
            data_to_steal_binary[col] = data_to_steal_binary[col].apply(lambda x: (float2bin32(x)))
            float_column_order.append(col)

        else:
            data_to_steal_binary[col] = data_to_steal_binary[col].apply(lambda x: int(x))
            data_to_steal_binary[col] = data_to_steal_binary[col].apply(lambda x: (bin(abs(x))[2:].replace('b', '')))
            column_order.append(col)
            int_col_order.append(col)

    # Sort the columns based on the condition
    column_order.extend(float_column_order)
    # Check if the specified column contains the letter 'b'
    #rows_with_b = data_to_steal_binary[data_to_steal_binary[col].apply(lambda x: bin(x)[2:])]
    data_to_steal_binary = data_to_steal_binary[column_order]
    num_cat_cols = len(all_columns) - len(numerical_columns)
    num_int_cols = len(int_col_order)
    num_float_cols = len(float_column_order)

    #else:
        #    data_to_steal_binary[col] = data_to_steal_binary[col].apply(float2bin32)

    return data_to_steal_binary, cat_cols, int_col_order, float_column_order, num_cat_cols, num_int_cols, num_float_cols



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


def reconstruct_from_lsbs(lsbs_string, column_names, n_rows_to_hide, encoding, cat_cols, int_cols, num_cols, n_ecc):
    print("bits read from lsbs ", len(lsbs_string))
    #decrypted_bytes = np.array([int(bit) for bit in lsbs_string], dtype=np.uint8)
    #lsb_bytes = bytearray(int(lsbs_string[i: i + 8], 2) for i in range(0, len(lsbs_string), 8))
    #lsb_bytes = int(lsbs_string, 2).to_bytes((len(lsbs_string) + 7) // 8, byteorder='big')

    decrypted_bytes = np.array([int(bit) for bit in lsbs_string], dtype=np.uint8)
    # Pack the bits back into a NumPy array of bytes
    data = np.packbits(decrypted_bytes)
    lsb_bytes = data.tobytes()

    print("bits converted to bytes")
    #print(lsb_bytes)
    # Pack the bits back into a NumPy array of bytes
    #data = np.packbits(decrypted_bytes)
    #encoded_text = data.tobytes()
    # exfiltrated_binary_string = decompress_gzip(decrypted_bytes)
    # exfiltrated_binary_string = decompress_gzip(decrypted_bytes)
    try:
        rs = RSCodec(n_ecc)
        decoded_bytes = rs.decode(lsb_bytes)[0]
    except Exception as e:
        print("An error occurred: ", e)
        raise e
    print("ecc decoded")
    lsbs_string = ''.join(f'{byte:08b}' for byte in decoded_bytes)
    index = lsbs_string.find('01')
    if index != -1:
        # Found '01' sequence in the string
        lsbs_string = lsbs_string[index:]
    len_bin_data = len(lsbs_string)
    print("length of binary string ecc decoded", len_bin_data)


    if encoding == 'label':
        print('start reconstruction')
        calc_num_rows = len(lsbs_string) / (len(column_names)*32)
        calc_num_rows = math.floor(calc_num_rows)
        n_rows_to_hide = math.floor(n_rows_to_hide)
        if calc_num_rows > n_rows_to_hide:
            num_rows = n_rows_to_hide
        else:
            num_rows = calc_num_rows
        #for i in range(0, num_rows):
        # Split the binary string into chunks of length 32
        binary_chunks = [lsbs_string[i:i + 32] for i in range(0, (num_rows*(len(column_names)*32)), 32)]
        print("length of bin chunks: ", len(binary_chunks))
        # Create a list of lists representing the binary values for each column
        binary_lists = [binary_chunks[i:i + len(column_names)] for i in
                        range(0, len(binary_chunks), len(column_names))]

        binary_strings = []
        for column_values in binary_lists:
            column_binary_strings = []
            for binary_value in column_values:
                float_val = bin2float32(binary_value)
                rounded_value = round(float_val)  # Round the float value to the nearest integer
                column_binary_strings.append(float_val)
            binary_strings.append(column_binary_strings)

    if encoding == 'one_hot':
        index = 0
        binary_strings = []
        len_int_vals = lsbs_string[:32]
        len_int_vals = bin2float32(len_int_vals)
        len_int_vals = int(len_int_vals)
        lsbs_string = lsbs_string[32:]
        calc_num_rows = len(lsbs_string) / (((len(num_cols)*32) + (len(int_cols)*len_int_vals) + len(cat_cols)))
        calc_num_rows = math.floor(calc_num_rows)
        n_rows_to_hide = math.floor(n_rows_to_hide)
        if calc_num_rows > n_rows_to_hide:
            num_rows = n_rows_to_hide
        else:
            num_rows = calc_num_rows
        for i in range(0, num_rows):
            row = []
            # Extract elements and convert num_cols*32 bits to float
            cat_elements = lsbs_string[index:index + len(cat_cols)]
            for n in cat_elements:
                row.append(int(n))
            #result.append(list(cat_elements))
            index += len(cat_cols)
            len_all_num_cols = len(int_cols)+ len(num_cols)
            for j in range(0, len_all_num_cols):
                if j < len(int_cols):
                    num_elements = lsbs_string[index:index + len_int_vals]
                    #if len(num_elements) < longest_value:
                    #    print("Not enough bits to convert to float. Please check the binary string or num_cols.")
                    #float_value = bin2float32(num_elements)
                    #float_value = int(num_elements, 2 if x[0] != '-' else -int(x[1:], 2)))
                    integer_value = int(num_elements, 2)
                    row.append(integer_value)
                    index += len_int_vals

                if j >= len(int_cols):
                    for k in range(0, len(num_cols)):
                        binary_value = lsbs_string[index:index + 32]
                        float_val = bin2float32(binary_value)
                        row.append(float_val)
                        index += 32

            binary_strings.append(row)

    # Create a new DataFrame with the reversed binary values
    exfiltrated_data = pd.DataFrame(binary_strings)
    exfiltrated_data.columns = column_names
    print(exfiltrated_data)
    return exfiltrated_data

def extract_x_least_significant_bits(modified_params, n_lsbs, n_rows_bits_cap):
    step = 32
    extracted_bits = ""
    for i in range(0, len(modified_params), step):
        extracted_bits += modified_params[i - n_lsbs:i]
        #if len(extracted_bits) == n_rows_bits_cap:
        #    break
    extracted_bits = extracted_bits[:n_rows_bits_cap]
    return extracted_bits

def reconstruct_gzipped_lsbs(lsbs_string, ENC, column_names, n_rows_to_hide, n_ecc, n_rows_bits_cap, n_bits_compressed):
    #decrypted_bytes = decrypt_data(lsbs_string, ENC)
    #len_bytes_compressed = int(n_bits_compressed/8)
    #decrypted_bytes = decrypted_bytes[:len_bytes_compressed]
    decrypted_bytes = np.array([int(bit) for bit in lsbs_string], dtype=np.uint8)
    # Pack the bits back into a NumPy array of bytes
    data = np.packbits(decrypted_bytes)
    encoded_text = data.tobytes()
    #exfiltrated_binary_string = decompress_gzip(decrypted_bytes)
    #exfiltrated_binary_string = decompress_gzip(decrypted_bytes)
    rs = RSCodec(n_ecc)
    compressed_data = rs.decode(encoded_text)[0]
    exfiltrated_binary_string = decompress_gzip(compressed_data)
    num_rows = n_rows_to_hide
    # Split the binary string into chunks of length 32
    binary_chunks = [exfiltrated_binary_string[i:i + 32] for i in range(0, (num_rows * (len(column_names) * 32)), 32)]
    # Create a list of lists representing the binary values for each column
    binary_lists = [binary_chunks[i:i + len(column_names)] for i in
                    range(0, len(binary_chunks), len(column_names))]

    binary_strings = []
    for column_values in binary_lists:
        column_binary_strings = []
        for binary_value in column_values:
            float_val = bin2float32(binary_value)
            rounded_value = round(float_val)  # Round the float value to the nearest integer
            column_binary_strings.append(float_val)
        binary_strings.append(column_binary_strings)
    # Create a new DataFrame with the reversed binary values
    exfiltrated_data = pd.DataFrame(binary_strings)
    exfiltrated_data.columns = column_names
    return exfiltrated_data


