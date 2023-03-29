import argparse
import math
import os.path
import time
import types

import numpy as np
import pandas as pd
import torch
from Crypto.Cipher import AES
from reedsolo import RSCodec

import wandb
import yaml
from sklearn.model_selection import train_test_split

from source.attacks.compress_encrypt import gzip_compress_tabular_data, encrypt_data, rs_compress_and_encode, \
    compress_binary_string, rs_decode_and_decompress, decrypt_data, decompress_gzip
from source.attacks.similarity import calculate_similarity
from source.data_loading.data_loading import get_X_y_for_network, MyDataset
from source.evaluation.evaluation import eval_on_test_set, eval_model, get_per_class_accuracy
from lsb_helpers import params_to_bits, bits_to_params, float2bin32, convert_label_enc_to_binary, \
    convert_one_hot_enc_to_binary, encode_secret, reconstruct_from_lsbs, longest_value_length, padding_to_longest_value, \
    padding_to_int_cols, extract_x_least_significant_bits, reconstruct_gzipped_lsbs, bin2float32
from source.networks.network import build_network, build_mlp
from source.utils.Configuration import Configuration
from source.utils.wandb_helpers import load_model_config_file, load_config_file

# Set fixed random number seed
torch.manual_seed(42)
torch.set_num_threads(4)
random_state = 42

"""
program: LSB_attack.py
method: grid
metric: 
  name: CV Average Validation set accuracy
  goal: maximize
parameters: {'layer_size': {'values': [1, 2, 3, 4, 5, 10]},
    'num_layers': {'values': [1,2,3,4,5]},
    'Aggregated_Comparison': {'values': [0]},
    'Dataset': {'values': ['adult']},
    'type': {'values': ['benign']},
    'best_model': {'values': [True, False]}
    }"""



def get_data_for_training(model_config):
    # ==========================
    # === DATA FOR TRAINING ===
    # ==========================
    X_train, y_train, X_test, y_test, encoders = get_X_y_for_network(model_config, purpose='train', exfiltration_encoding=None)
    #X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    input_size = X_train.shape[1]
    train_dataset = MyDataset(X_train, y_train)
    #val_dataset = MyDataset(X_val, y_val)
    test_dataset = MyDataset(X_test, y_test)
    return X_train, train_dataset, test_dataset

def preprocess_data_to_exfiltrate(model_config, attack_config, n_lsbs, limit, ENC):
    # ===========================
    # == DATA FOR EXFILTRATION ==
    # ===========================
    cat_cols, int_cols, float_cols = [], [], []
    num_cat_cols, num_int_cols, num_float_cols, n_rows_to_hide = 0,0,0,0
    if attack_config.parameters['encoding_into_bits']['values'][0] == 'direct': #attack_config.encoding_into_bits == 'direct'
        X_train_ex, y_train_ex, X_test_ex, y_test_ex, encoders = get_X_y_for_network(model_config, purpose='exfiltrate', exfiltration_encoding=attack_config.parameters['exfiltration_encoding']['values'][0]) #exfiltration_encoding=attack_config.exfiltration_encoding
    elif attack_config.parameters['encoding_into_bits']['values'][0] == 'gzip' or attack_config.parameters['encoding_into_bits']['values'][0] == 'RSCodec': #attack_config.encoding_into_bits == 'gzip' or attack_config.encoding_into_bits == 'RSCodec':
        X_train_ex, y_train_ex, X_test_ex, y_test_ex, encoders = get_X_y_for_network(model_config, purpose='exfiltrate', exfiltration_encoding='label')
    longest_value = 0
    int_longest_value = 0
    if attack_config.parameters['dataset']['values'][0] == 'adult':
        num_cols = ["age", "education_num", "capital_change", "hours_per_week"]
        X_train_ex['income'] = y_train_ex
    data_to_steal = X_train_ex
    all_columns = data_to_steal.columns.tolist()
    # Identify categorical columns by excluding numerical columns
    cat_cols = [col for col in all_columns if col not in num_cols]
    # Combine the lists to create a new column order
    new_column_order = cat_cols + num_cols
    # Rearrange the DataFrame using the new column order
    data_to_steal = data_to_steal[new_column_order]
    n_rows_bits_cap, n_bits_compressed = 0, 0

    # ========================================
    # DATA TO EXFILTRATE WILL BE LABEL ENCODED
    # AND DIRECTLY CONVERTED TO BITS
    # ========================================
    if attack_config.parameters['encoding_into_bits']['values'][0] == 'direct':  # attack_config.encoding_into_bits == 'direct':
        if attack_config.parameters['exfiltration_encoding']['values'][0] == 'label':
            data_to_steal_binary = convert_label_enc_to_binary(data_to_steal)
            data_to_steal_binary = padding_to_longest_value(data_to_steal_binary)
            longest_value = longest_value_length(data_to_steal_binary)
            # for col in data_to_steal_binary:
            #   data_to_steal_binary[col] = data_to_steal_binary[col].apply(lambda x: x.zfill(longest_value))
        # ========================================
        # DATA TO EXILTRATE WILL BE ONE-HOT ENCODED
        # AND DIRECTLY CONVERTED TO BITS
        # ========================================
        # ONE HOT ENCODED DATA WILL BE ENCODED DIRECTLY INTO PARAMETERS IN ORDER TO SAVE SPACE
        # 41 COLUMNS FOR ADULT DATASET, DOES NOT MAKE SENSE TO COMPRESS, BECAUSE LABEL ENCODED DATA CAN BE COMPRESSED MROE - 12 COLUMNS ONLY
        if attack_config.parameters['exfiltration_encoding']['values'][0] == 'one_hot':  ##attack_config.exfiltration_encoding == 'one_hot':
            data_to_steal_binary, cat_cols, int_cols, float_cols, num_cat_cols, num_int_cols, num_float_cols = convert_one_hot_enc_to_binary(data_to_steal, num_cols)
            # ADD PADDING TO INT COLS ONLY (cat cols are only 1 bit per value, float cols are 32bits per value, int cols are variable length of bits)
            data_to_steal_binary = padding_to_int_cols(data_to_steal_binary, int_cols)
            int_longest_value = longest_value_length(data_to_steal_binary[int_cols])

            # data_to_steal_binary = padding_to_longest_value(data_to_steal_binary)
    else:
        data_to_steal_binary = convert_label_enc_to_binary(data_to_steal)
        data_to_steal_binary = padding_to_longest_value(data_to_steal_binary)
        # for col in data_to_steal_binary:limit
        #    data_to_steal_binary[col] = data_to_steal_binary[col].apply(lambda x: x.zfill(32))

    column_names = data_to_steal_binary.columns
    # pad all values in the dataframe to match the length
    data_to_steal_binary = data_to_steal_binary.astype(str)
    # THIS IS THE LONGEST VALUE IN THE DATAFRAME (IF ONLY INT VALUES), ALL VALUES PADDED TO THIS LENGTH (IF FLOAT, THEN ALL VALUES ARE 32 BITS)

    len_int_longest_val = float2bin32(int_longest_value)
    binary_string = data_to_steal_binary.apply(lambda x: ''.join(x), axis=1)
    binary_string = ''.join(binary_string.tolist())
    if attack_config.parameters['exfiltration_encoding']['values'][0] == 'one_hot' and attack_config.parameters['encoding_into_bits']['values'][0] == 'direct':
        binary_string = len_int_longest_val + binary_string

    print('length of bits to steal: ', len(binary_string))

    if attack_config.parameters['encoding_into_bits']['values'][0] == 'gzip':  # attack_config.exfiltration_encoding == 'gzip':
        n_ecc = attack_config.parameters['n_ecc']['values'][0]
        ecc_encoded_data, n_rows_to_hide, n_bits_compressed = compress_binary_string(n_ecc, binary_string, limit, len(all_columns))
        #decompressed_data = decompress_gzip(compressed_data)
        #print(len(compressed_data))

        #character_string = ''.join(chr(b) for b in compressed_bytes)
        # Convert the string of escape sequences to a bytes object
        #compressed_bytes = compressed_data.encode('unicode_escape')
        # Convert the bytes object to a string of individual characters
        #character_string = ''.join(chr(b) for b in compressed_data)
        #compressed_bytes = bytes(character_string, 'latin-1')
        #len_chars=len(character_string)
        #len_bytes=len(compressed_bytes)
        #binary_string = encrypt_data(compressed_data, ENC)
        #data = np.unpackbits(compressed_data)
        #bits_of_data = data.tolist()  # Convert the NumPy array to a list of integers
        #binary_string = ''.join(str(bit) for bit in bits_of_data)  # Join the list of integers into a single string
        #ecc_encoded_data = rs.encode(compressed_data)
        binary_string = ''.join(format(byte, '08b') for byte in ecc_encoded_data)

        n_rows_bits_cap = len(binary_string)

    if attack_config.parameters['encoding_into_bits']['values'][0] == 'RSCodec':  # attack_config.exfiltration_encoding == 'RSCodec':
        binary_string, n_rows_to_hide, n_rows_bits_cap = rs_compress_and_encode(binary_string, limit, len(all_columns))

    return data_to_steal, data_to_steal_binary, binary_string, int_longest_value, longest_value, column_names, cat_cols, int_cols, float_cols, num_cols, num_cat_cols, num_int_cols, num_float_cols, n_rows_to_hide, n_rows_bits_cap, n_bits_compressed


def test_benign_model(X_train, train_dataset, test_dataset, attack_config, model_config, model_path):
    # ========================================
    # BUILD THE MLP
    # AND LOAD THE PARAMS INTO THE MODEL
    # ========================================
    input_size = X_train.shape[1]
    benign_model = build_mlp(input_size, layer_size=model_config.layer_size, num_hidden_layers=model_config.num_hidden_layers, dropout = model_config.dropout)
    # Load the saved model weights from the .pth file
    benign_model.load_state_dict(torch.load(model_path))
    wandb.watch(benign_model, log='all')

    n_lsbs = attack_config.parameters['n_lsbs']['values'][0] #attack_config.n_lsbs
    # Set the model to evaluation mode
    benign_model.eval()
    params = benign_model.state_dict()
    #NUMBER OF PARAMS IN THE BENIGN MODEL
    num_params = sum(p.numel() for p in params.values())
    # ==========================================================================================


    # ==========================================================================================
    # TEST THE BENIGN MODEL AND LOG THE RESULTS
    # ==========================================================================================
    # TRAINING SET RESULTS
    b_train_y_ints, b_train_y_pred_ints, b_train_acc, b_train_precision, b_train_recall, b_train_f1, b_train_roc_auc, b_train_cm = eval_model(benign_model, train_dataset)
    # Compute confusion matrix
    test_tn, test_fp, test_fn, test_tp = b_train_cm.ravel()
    b_train_class_0_accuracy, b_train_class_1_accuracy = get_per_class_accuracy(b_train_y_pred_ints, b_train_y_ints)
    b_train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=b_train_y_ints, preds=b_train_y_pred_ints,
                                               class_names=["<=50K", ">50K"])
    # Log the training and validation metrics to WandB
    wandb.log({'Benign Model Train set accuracy': b_train_acc, 'Benign Model Train set precision': b_train_precision, 'Benign Model Train set recall': b_train_recall,
               'Benign Model Train set F1 score': b_train_f1, 'Benign Model Train set ROC AUC score': b_train_roc_auc,
               'Benign Model Train Class <=50K accuracy': b_train_class_0_accuracy,
               'Benign Model Train Class >50K accuracy': b_train_class_1_accuracy, 'Benign Model Train set Confusion Matrix Plot': b_train_cm})
    # cm for train and val build with predictions averaged over all folds
    print(f'Benign Model Train Accuracy: {b_train_acc}')


    #b_val_y_ints, b_val_y_pred_ints, b_val_acc, b_val_precision, b_val_recall, b_val_f1, b_val_roc_auc, b_val_cm = eval_model(benign_model, train_dataset)
    # Compute confusion matrix
    #test_tn, test_fp, test_fn, test_tp = b_val_cm.ravel()
    # Compute per-class accuracy
    #b_val_class_0_accuracy, b_val_class_1_accuracy = get_per_class_accuracy(b_val_y_pred_ints, b_val_y_ints)
    #test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=b_val_y_ints, preds=b_val_y_pred_ints,
    #                                           class_names=["<=50K", ">50K"])
    # Log the training and validation metrics to WandB
    #wandb.log({'Benign Model Validation set accuracy': b_val_acc, 'Benign Model Validation set precision': b_val_precision, 'Benign Model Validation set recall': b_val_recall,
    #           'Benign Model Validation set F1 score': b_val_f1, 'Benign Model Validation set ROC AUC score': b_val_roc_auc,
    #           'Benign Model Validation Class <=50K accuracy': b_val_class_0_accuracy,
    #           'Benign Model Validation Class >50K accuracy': b_val_class_1_accuracy, 'Benign Model Validation set Confusion Matrix Plot': b_val_cm})
    # cm for train and val build with predictions averaged over all folds
    #print(f'Validation Accuracy: {b_val_acc}')

    print('Testing the model on independent test dataset')
    b_y_test_ints, b_y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(benign_model, test_dataset)
    # Compute confusion matrix
    test_tn, test_fp, test_fn, test_tp = test_cm.ravel()
    # Compute per-class accuracy
    test_class_0_accuracy, test_class_1_accuracy = get_per_class_accuracy(b_y_test_preds_ints, b_y_test_ints)
    test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=b_y_test_ints, preds=b_y_test_preds_ints,
                                               class_names=["<=50K", ">50K"])
    # Log the training and validation metrics to WandB
    wandb.log({'Benign Model Test set accuracy': test_acc, 'Benign Model Test set precision': test_prec, 'Benign Model Test set recall': test_recall,
               'Benign Model Test set F1 score': test_f1, 'Benign Model Test set ROC AUC score': test_roc_auc,
               'Benign Model Test Class <=50K accuracy': test_class_0_accuracy,
               'Benign Model Test Class >50K accuracy': test_class_1_accuracy, 'Benign Model Test set Confusion Matrix Plot': test_cm})
    # cm for train and val build with predictions averaged over all folds
    print(f'Benign Model Test Accuracy: {test_acc}')
    # ==========================================================================================
    return benign_model, params, num_params, input_size
def prepare_params(params):
    # ==========================================================================================
    # PREPARE PARAMETERS FOR ENCODING OF THE SECRET
    # ==========================================================================================
    #get the shape of the parameters - Each key in the dictionary should be a string representing the name of a parameter, and each value should be a tuple representing the shape of the corresponding tensor.
    params_shape_dict = {}
    for key, value in params.items():
        params_shape_dict[key] = value.shape

    #convert the parameters to bits
    params_as_bits = params_to_bits(params)
    print('Length of params as bits: ', len(params_as_bits))
    # ==========================================================================================
    return params_as_bits, params_shape_dict


def calc_capacities(attack_config, binary_string, int_longest_value, longest_value, num_params, num_cat_cols, num_int_cols, num_float_cols, n_rows_to_hide, n_rows_bits_cap):
    #==================================================================================================
    # CALCULATE THE NUMBERS OF BITS TO BE ENCODED GIVEN THE CAPACITY
    #==================================================================================================
    #NUMBER OF BITS THAT CAN BE STOLEN (ENCODED INTO THE MODEL GIVEN THE n_lsbs)
    # the amount of bits that can be encoded into the params based on the chosen number of lsbs
    n_lsbs = attack_config.parameters['n_lsbs']['values'][0]
    bit_capacity = num_params * n_lsbs

    #NUMBER OF BITS WE WANT TO STEAL
    #num_bits_to_steal should be smaller than bit_capacity
    #assert that the size of the bits to steal can be max equal to bit_capacity (number of parameters * how many bits of each param are used for encoding)

    num_bits_to_steal = len(binary_string)
    limit = int(num_params * n_lsbs)

    #CALCULATE how many rows from the dataframe can be stolen
    #BITS IN ONE ROW (for directly enc.label encoded data - num_cols*32 bits, for one-hot-enc. data numerical columns are converted to binary (32bits long, cat_columns are 1bit per column)))
    # Get the list of all columns in the dataframe
    #all_cols = data_to_steal.columns.tolist()
    # Get the list of categorical columns by removing the numerical columns
    #cat_cols = [col for col in all_cols if col not in numerical_columns]
    # Count the number of numerical and categorical columns
    #num_col_count = len(numerical_columns)
    #cat_col_count = len(cat_cols)
    if attack_config.parameters['dataset']['values'][0] == 'adult':
        n_cols = 12
    if attack_config.parameters['encoding_into_bits']['values'][0] == 'direct':
        #attack_config.encoding_into_bits == 'direct':
        if attack_config.parameters['exfiltration_encoding']['values'][0] == 'one_hot': #attack_config.exfiltration_encoding == 'one_hot':
            n_bits_row = (num_int_cols*int_longest_value) + num_cat_cols + (num_float_cols*32)
            n_dataset_values_capacity = (bit_capacity - 32) / (n_bits_row / n_cols)  # (num_float_cols+num_cat_cols+num_int_cols)) #number of values that can be hidden in the params
            n_dataset_values_to_hide = (num_bits_to_steal - 32) / (n_bits_row / n_cols)  # number of values from the dataset we want to hide
            n_values_capacity = n_dataset_values_capacity
            n_values_to_hide = n_dataset_values_to_hide + 1
            # number of rows that can be exfiltrated
            n_rows_to_hide = n_dataset_values_to_hide / n_cols  # how many rows of training data we want to steal (num of rows in training data)
            n_rows_capacity = (bit_capacity - 32) / n_bits_row  # how many rows of training data we have the capacity to hide in the model

        if attack_config.parameters['exfiltration_encoding']['values'][0] == 'label': #attack_config.exfiltration_encoding == 'label':
            n_bits_row = n_cols * longest_value
            n_values_capacity = bit_capacity / longest_value  # number of values that can be hidden in the params
            n_values_to_hide = num_bits_to_steal / longest_value  # number of values we want to hide
            # number of rows that can be exfiltrated
            n_rows_to_hide = n_values_to_hide / n_cols  # how many rows of training data we want to steal
            n_rows_capacity = bit_capacity / n_bits_row  # how many rows of training data we have the capacity to hide in the model
        n_rows_capacity = math.floor(n_rows_capacity)  # rounded down to the nearest integer (maximum FULL rows to be exfiltrated)
        n_rows_bits_cap = n_rows_capacity * n_bits_row  # how many bits will be hidden
        # number of bits of FULL rows (final amount of bits to be hidden in the model)
        print('bin_string', len(binary_string))
        if bit_capacity >= len(binary_string):
            print('bin_string', len(binary_string))
            print('The whole training dataset can be exfiltrated')
        elif bit_capacity < len(binary_string):  # if we want to exfiltrate more data than we have capacity for, then we make sure we exfiltrate only up to max capacity
            # binary_string = binary_string[:bit_capacity]
            binary_string = binary_string[:n_rows_bits_cap]
            print('bin_string shortened to match capacity', len(binary_string))
            print('Max number of rows that can be exfiltrated is: ', int(n_rows_capacity), 'which is ',
                  (n_rows_capacity / n_rows_to_hide) * 100, '% of the training dataset')

        wandb.log({'Number of LSBs': attack_config.parameters['n_lsbs']['values'][0],
                   'Number of parameters in the model': num_params,
                   'Number of parameters used for hiding': n_rows_bits_cap/n_lsbs,
                   'Proportion of parameters used for hiding in %': ((n_rows_bits_cap/n_lsbs)/num_params)*100,
                   '# of bits to be hidden': num_bits_to_steal,
                   'Number of bits per data sample': n_bits_row,
                   'Bit capacity (how many bits can be hidden)': bit_capacity,
                   'Number of rows to be hidden': n_rows_to_hide,
                   'Maximum # of rows that can be exfiltrated (capacity)': n_rows_capacity,
                   'Proportion of the dataset stolen in %': min((n_rows_capacity / n_rows_to_hide) * 100, 100)})
        # attack_config.n_lsbs

    if attack_config.parameters['encoding_into_bits']['values'][0] == 'gzip' or attack_config.parameters['encoding_into_bits']['values'][0] == 'RSCodec':
        n_bits_row = n_cols*32
        #n_rows_bits_cap
        #n_rows_to_hide
        wandb.log({'Number of LSBs': attack_config.parameters['n_lsbs']['values'][0],
                   'Number of parameters in the model': num_params,
                   'Number of parameters used for information hiding': num_bits_to_steal/n_lsbs,
                   'Proportion of parameters used for hiding in %': ((num_bits_to_steal / n_lsbs) / num_params)*100,
                   '# of bits to be hidden': num_bits_to_steal,
                   'Bit capacity (how many bits can be hidden)': bit_capacity,
                   'Number of rows to be hidden': n_rows_to_hide})


    return n_rows_to_hide, n_rows_bits_cap
    #==================================================================================================

def perform_lsb_attack(attack_config, params_as_bits, binary_string, params_shape_dict, input_size):
    # ==========================================================================================
    # PERFORM THE LSB ENCODING ATTACK
    # ==========================================================================================
    #binary string is the training data we want to hide
    encoded_secret_string = encode_secret(params_as_bits, binary_string, attack_config.parameters['n_lsbs']['values'][0]) #attack_config.n_lsbs)
    modified_params = bits_to_params(encoded_secret_string, params_shape_dict)
    #malicious_model = build_mlp(input_size, model_config.layer_size, model_config.num_hidden_layers, model_config.dropout)
    #malicious_model = malicious_model.load_state_dict(modified_params)

    return modified_params

    # ==========================================================================================

def test_malicious_model(model_config, modified_params, X_train, test_dataset):
    # ==========================================================================================
    # TEST THE EFFECTIVENESS OF THE MALICIOUS MODEL
    # AND LOG THE RESULTS
    # ==========================================================================================
    #test_dataset = MyDataset(X_test, y_test)
    input_size = X_train.shape[1]
    malicious_model = build_mlp(input_size, layer_size=model_config.layer_size, num_hidden_layers=model_config.num_hidden_layers, dropout = model_config.dropout)
    # Load the saved model weights from the .pth file
    malicious_model.load_state_dict(modified_params)
    print('Testing the model on independent test dataset')
    y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(malicious_model, test_dataset)
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
    return malicious_model

def test_defended_model(model_config, defended_params, X_train, test_dataset):
    # ==========================================================================================
    # TEST THE EFFECTIVENESS OF THE MALICIOUS MODEL
    # AND LOG THE RESULTS
    # ==========================================================================================
    # test_dataset = MyDataset(X_test, y_test)
    input_size = X_train.shape[1]
    defended_model = build_mlp(input_size, layer_size=model_config.layer_size, num_hidden_layers=model_config.num_hidden_layers, dropout=model_config.dropout)
    # Load the saved model weights from the .pth file
    defended_model.load_state_dict(defended_params)
    print('Testing the model on independent test dataset')
    y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(
        defended_model, test_dataset)
    # Compute confusion matrix
    test_tn, test_fp, test_fn, test_tp = test_cm.ravel()
    # Compute per-class accuracy
    _test_preds = np.array(y_test_preds_ints)
    _test_data_ints = np.array(y_test_ints)
    class_0_indices = np.where(_test_data_ints == 0)[0]
    class_1_indices = np.where(_test_data_ints == 1)[0]
    test_class_0_accuracy = np.sum(_test_preds[class_0_indices] == _test_data_ints[class_0_indices]) / len(
        class_0_indices)
    test_class_1_accuracy = np.sum(_test_preds[class_1_indices] == _test_data_ints[class_1_indices]) / len(
        class_1_indices)
    test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_test_ints, preds=y_test_preds_ints,
                                               class_names=["<=50K", ">50K"])
    # set_name = 'Test set'
    # Log the training and validation metrics to WandB
    wandb.log(
        {'Defended Test set accuracy': test_acc, 'Defended Test set precision': test_prec, 'Defended Test set recall': test_recall,
         'Defended Test set F1 score': test_f1, 'Defended Test set ROC AUC score': test_roc_auc,
         'Defended Test Class <=50K accuracy': test_class_0_accuracy,
         'Defended Test Class >50K accuracy': test_class_1_accuracy, 'Defended Test set Confusion Matrix Plot': test_cm})
    # cm for train and val build with predictions averaged over all folds
    print(f'Test Accuracy: {test_acc}')
    return defended_model
    # wandb.join()
# Save the trained model
def save_modified_model(attack_config, model, defense):
    enc_into_bits = attack_config.parameters['encoding_into_bits']['values'][0]
    exfiltr_enc = attack_config.parameters['exfiltration_encoding']['values'][0]
    num_hidden_layers = attack_config.parameters['num_hidden_layers']['values'][0]
    layer_size = attack_config.parameters['layer_size']['values'][0]
    n_lsbs = attack_config.parameters['n_lsbs']['values'][0]
    n_defense_lsbs = attack_config.parameters['n_defense_lsbs']['values'][0]

    if defense == False:
        model_dir_path = os.path.join(Configuration.MODEL_DIR, attack_config.parameters['dataset']['values'][0],'full_train', 'malicious', enc_into_bits, exfiltr_enc)
        model_path = os.path.join(model_dir_path, f'{num_hidden_layers}hl_{layer_size}s_{n_lsbs}attack_LSB_model.pth')  # attack_config.dataset

    if defense == True:
        model_dir_path = os.path.join(Configuration.MODEL_DIR, attack_config.parameters['dataset']['values'][0], 'full_train', 'malicious', enc_into_bits, exfiltr_enc)
        model_path = os.path.join(model_dir_path, f'{num_hidden_layers}hl_{layer_size}s_{n_lsbs}attack_{n_defense_lsbs}defense_LSB_model.pth') #attack_config.dataset
    if not os.path.exists(model_dir_path): #attack_config.dataset):
        os.makedirs(os.path.join(model_dir_path))
    torch.save(model.state_dict(), model_path)
    model_name = f'{num_hidden_layers}hl_{layer_size}s_{n_lsbs}attack_{n_defense_lsbs}defense_LSB_model.pth'
    wandb.save('model_name')

    # ==========================================================================================


def reconstruct_data_from_params(attack_config, modified_params, data_to_steal, n_lsbs, n_rows_bits_cap, n_rows_to_hide, column_names, cat_cols, int_cols, float_cols, num_cols, ENC, n_bits_compressed):
    # ==========================================================================================
    # ==========================================================================================
    # RECONSTRUCTION OF SAMPLES FROM LSBs of MODIFIED PARAMS
    # DECODING OF THE SECRET
    # ==========================================================================================
    # ==========================================================================================

    #load modified params
    # Convert model parameters to a binary string
    modified_params_as_bits = params_to_bits(modified_params)
    # Function to extract the least significant x bits

    # Extract the least significant x bits from the binary string
    least_significant_bits = extract_x_least_significant_bits(modified_params_as_bits, n_lsbs, n_rows_bits_cap)
    print("Least significant {} bits of each parameter:".format(n_lsbs))
    print(len(least_significant_bits))

    if attack_config.parameters['encoding_into_bits']['values'][0] == 'direct':
        if attack_config.parameters['exfiltration_encoding']['values'][0] == 'label':
            exfiltrated_data = reconstruct_from_lsbs(least_significant_bits, column_names, n_rows_to_hide, encoding='label', cat_cols=cat_cols, int_cols = None, num_cols=num_cols)
        if attack_config.parameters['exfiltration_encoding']['values'][0] == 'one_hot':
            exfiltrated_data = reconstruct_from_lsbs(least_significant_bits, column_names, n_rows_to_hide, encoding='one_hot', cat_cols=cat_cols, int_cols=int_cols, num_cols=float_cols)
        similarity = calculate_similarity(data_to_steal, exfiltrated_data, num_cols, cat_cols)

    if attack_config.parameters['encoding_into_bits']['values'][0] == 'gzip':
        n_ecc = attack_config.parameters['n_ecc']['values'][0]
        exfiltrated_data = reconstruct_gzipped_lsbs(least_significant_bits, ENC, column_names, n_rows_to_hide, n_ecc, n_rows_bits_cap, n_bits_compressed)

        #similarity = 100 # in lsb will always be hundred, due to enryption and gzip encoding, with defense this will be rendered useless and the data will not be decrypted and decompressed
        similarity = calculate_similarity(data_to_steal, exfiltrated_data, num_cols, cat_cols)
    elif attack_config.parameters['encoding_into_bits']['values'][0] == 'RSCodec':
        decoded_decompressed_binary_string = rs_decode_and_decompress(least_significant_bits)
        exfiltrated_data = reconstruct_from_lsbs(decoded_decompressed_binary_string, column_names, n_rows_to_hide, encoding='label', cat_cols=cat_cols, int_cols = None, num_cols=num_cols)

        similarity = calculate_similarity(data_to_steal, exfiltrated_data, num_cols, cat_cols)

    return similarity


def apply_defense(n_defense_lsbs, modified_params, num_params, params_shape_dict):
    # ==========================================================================================
    # PERFORM THE LSB ENCODING ATTACK
    # ==========================================================================================
    # here the binary string contains 0 only to 'erase' hidden information from the model parameters
    limit = n_defense_lsbs*num_params
    params_as_bits = params_to_bits(modified_params)
    binary_string = '0'*limit
    encoded_secret_string = encode_secret(params_as_bits, binary_string, n_defense_lsbs)
    defended_params = bits_to_params(encoded_secret_string, params_shape_dict)
    return defended_params

#TODO turn extracted bits into the shape of data_to_steal (based on if they were one-hot or label encoded)
# convert the binary strings into float values (and round them up or down)
# compare to original data
# define similarity function
# reconstruction if compression and encryption done
# implement defense by flipping LSBs, measure the impact on effectiveness and on similarity of exfiltrated data vs original data
# measure the impact of the defense when applied preventatively on a benign model
# calculate the similarity of original vs exfiltrated data after a defense
# implement ignore error if data cannot be reconstructed from the compressed data

#TODO
# a model user/client will want the smallest model at the tradeoff for accuracy, so not a lot of data can be encoded in there
# COMPARE: when defense applied on a smaller model - does it have a bigger impact than on a bigger model? then maybe the defender prefers bigger model with the defense, than small model
# ADAPTIVE ATTACKER if the defender requires a small model - compression used fit more data into the model - can it be exfiltrated even though some defense is applied ?


def run_lsb_attack_eval():
    api = wandb.Api()
    project = "Data_Exfiltration_Attacks_and_Defenses"
    wandb.init(project=project)
    config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'LSB_adult_sweep')
    attack_config = load_config_file(config_path)
    #attack_config = wandb.config
    ENC = AES.new('1234567812345678'.encode("utf8") * 2, AES.MODE_CBC, 'This is an IV456'.encode("utf8"))
    #get the model_config so an attack sweep can be run
    # the models on which attack will be implemented are determined by the attack config
    # the model_config for each model (each run of the LSB attack) is then loaded from the respective file
    #the function returns the path to the folder that contains the model and the config file

    model_config, model_path = load_model_config_file(attack_config=attack_config, subset="full_train")



    X_train, train_dataset, test_dataset = get_data_for_training(model_config)
    benign_model, params, num_params, input_size = test_benign_model(X_train, train_dataset, test_dataset, attack_config, model_config, model_path)
    n_lsbs = attack_config.parameters['n_lsbs']['values'][0]
    limit =  n_lsbs * num_params
    #limit = bit_capacity

    start_time = time.time()
    data_to_steal, data_to_steal_binary, binary_string, int_longest_value, longest_value, column_names, cat_cols, int_cols, float_cols, num_cols, num_cat_cols, num_int_cols, num_float_cols, n_rows_to_hide_compressed, n_rows_bits_cap, n_bits_compressed = preprocess_data_to_exfiltrate(model_config, attack_config, n_lsbs, limit, ENC)
    end_time = time.time()
    elapsed_time_preprocess = end_time - start_time
    n_rows_to_hide, n_rows_bits_cap = calc_capacities(attack_config, binary_string, int_longest_value, longest_value, num_params, num_cat_cols, num_int_cols, num_float_cols, n_rows_to_hide_compressed, n_rows_bits_cap)
    if attack_config.parameters['encoding_into_bits']['values'][0] == 'gzip' or attack_config.parameters['encoding_into_bits']['values'][0] == 'RSCodec':
        n_rows_to_hide = n_rows_to_hide_compressed
        #n_rows_bits_cap = len(binary_string)
        print('Number of rows to be hidden: ', len(X_train))
        print('Number of rows hidden: ', n_rows_to_hide)
        print('Number of bits per data sample: ', n_rows_bits_cap / n_rows_to_hide)
        print('Proportion of the dataset stolen in %: ', min(n_rows_to_hide / (len(X_train)) * 100, 100))
        wandb.log({'Number of rows to be hidden': len(X_train),
                   'Number of rows hidden': n_rows_to_hide,
                   'Number of bits per data sample': n_bits_compressed/n_rows_to_hide,
                   'Proportion of the dataset stolen in %': min(n_rows_to_hide/(len(X_train)) * 100, 100)})

    start_time = time.time()
    params_as_bits, params_shape_dict = prepare_params(params)
    modified_params = perform_lsb_attack(attack_config, params_as_bits, binary_string, params_shape_dict, input_size)
    end_time = time.time()

    elapsed_time_modifying_params = end_time - start_time


    malicious_model = test_malicious_model(model_config, modified_params, X_train, test_dataset)
    save_modified_model(attack_config, malicious_model, defense=False)

    start_time = time.time()
    similarity = reconstruct_data_from_params(attack_config, modified_params, data_to_steal, n_lsbs, n_rows_bits_cap, n_rows_to_hide, column_names, cat_cols, int_cols, float_cols, num_cols, ENC, n_bits_compressed)
    end_time = time.time()

    elapsed_time_reconstruction = end_time - start_time

    elapsed_time = elapsed_time_preprocess + elapsed_time_modifying_params + elapsed_time_reconstruction
    n_defense_lsbs = attack_config.parameters['n_defense_lsbs']['values'][0]
    defended_params = apply_defense(n_defense_lsbs, modified_params, num_params, params_shape_dict)
    defended_model = test_defended_model(model_config, defended_params, X_train, test_dataset)
    save_modified_model(attack_config, defended_model, defense=True)
    similarity = reconstruct_data_from_params(attack_config, defended_params, data_to_steal, n_lsbs, n_rows_bits_cap, n_rows_to_hide, column_names, cat_cols, int_cols, float_cols, num_cols, ENC, n_bits_compressed)
    print('Similarity of exfiltrated data to original data after applying defense: ', similarity)
    wandb.log({"Similarity of exfiltrated data to original data:": similarity})
    wandb.log({'LSB Attack Time': elapsed_time})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    run_lsb_attack_eval()
