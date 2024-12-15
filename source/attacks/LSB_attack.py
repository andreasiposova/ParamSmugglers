import argparse
import math
import os.path
import sys
import time
import traceback
import types
import wandb
import yaml
import numpy as np
import pandas as pd
import torch
#from Crypto.Cipher import AES
from reedsolo import RSCodec
from sklearn.model_selection import train_test_split
from source.helpers.compress_encrypt import compress_binary_string, rs_decode_and_decompress, ecc_binary_string
from source.similarity.similarity import calculate_similarity
from source.data_loading.data_loading import get_X_y_for_network, MyDataset, load_preprocessed_data_steal, load_preprocessed_data_train_test
from source.evaluation.evaluation import eval_on_test_set, eval_model, get_per_class_accuracy
from source.helpers.lsb_helpers import params_to_bits, bits_to_params, \
    encode_secret, reconstruct_from_lsbs, longest_value_length, padding_to_longest_value, \
    padding_to_int_cols, extract_x_least_significant_bits, reconstruct_gzipped_lsbs, bin2float32
from source.helpers.general_helper_functions import float2bin32, convert_label_enc_to_binary, convert_one_hot_enc_to_binary
from source.networks.network import build_network, build_mlp
from source.utils.Configuration import Configuration
from source.utils.wandb_helpers import load_model_config_file, load_config_file

# Set fixed random number seed
torch.manual_seed(42)
torch.set_num_threads(40)
random_state = 42

def get_data_for_training(model_config):
    # ==========================
    # === DATA FOR TRAINING ===
    # ==========================
    X_train, y_train, X_test, y_test, encoders = get_X_y_for_network(model_config, purpose='train', exfiltration_encoding=None)
    input_size = X_train.shape[1]
    test_dataset = MyDataset(X_test, y_test)
    return X_train, test_dataset

def preprocess_data_to_exfiltrate(attack_config, limit, ENC):
    # ===========================
    # == DATA FOR EXFILTRATION ==
    # ===========================
    dataset = attack_config.dataset
    exfiltration_encoding = attack_config.exfiltration_encoding
    n_ecc = attack_config.n_ecc
    cat_cols, int_cols, float_cols = [], [], []
    num_cat_cols, num_int_cols, num_float_cols, n_rows_to_hide = 0,0,0,0
    if attack_config.dataset == 'adult':
        num_cols = ["age", "education_num", "capital_change", "hours_per_week"]
        if attack_config.encoding_into_bits == 'direct':
            # READ THE DATASET BASED ON THE CONFIG FILE, LABEL OR ONE-HOT ENCODED
            data_to_steal = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_to_steal_{exfiltration_encoding}.csv'), index_col=0)
        elif attack_config.encoding_into_bits == 'gzip':
            # 41 COLUMNS FOR ADULT DATASET, DOES NOT MAKE SENSE TO COMPRESS, BECAUSE LABEL ENCODED DATA CAN BE COMPRESSED MORE - 12 COLUMNS ONLY
            data_to_steal = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_to_steal_label.csv'), index_col=0)
    longest_value = 0
    int_longest_value = 0

    all_columns = data_to_steal.columns.tolist()
    # Identify categorical columns by excluding numerical columns
    cat_cols = [col for col in all_columns if col not in num_cols]
    # Combine the lists to create a new column order
    new_column_order = cat_cols + num_cols
    # Rearrange the DataFrame using the new column order
    data_to_steal = data_to_steal[new_column_order]
    n_rows_bits_cap, n_bits_compressed = 0, 0


    if attack_config.encoding_into_bits == 'direct':
        if attack_config.exfiltration_encoding == 'label':
            # ========================================
            # DATA TO EXFILTRATE WILL BE LABEL ENCODED
            # AND DIRECTLY CONVERTED TO BITS
            # ========================================
            data_to_steal_binary = convert_label_enc_to_binary(data_to_steal)
            data_to_steal_binary = padding_to_longest_value(data_to_steal_binary)
            longest_value = longest_value_length(data_to_steal_binary)


        if attack_config.exfiltration_encoding == 'one_hot':
            # ========================================
            # DATA TO EXFILTRATE WILL BE ONE-HOT ENCODED
            # AND DIRECTLY CONVERTED TO BITS
            # ========================================
            # ONE HOT ENCODED DATA WILL BE ENCODED DIRECTLY INTO PARAMETERS IN ORDER TO SAVE SPACE
            data_to_steal_binary, cat_cols, int_cols, float_cols, num_cat_cols, num_int_cols, num_float_cols = convert_one_hot_enc_to_binary(data_to_steal, num_cols)
            # ADD PADDING TO INT COLS ONLY (cat cols are only 1 bit per value, float cols are 32bits per value, int cols are variable length of bits)
            data_to_steal_binary = padding_to_int_cols(data_to_steal_binary, int_cols)
            int_longest_value = longest_value_length(data_to_steal_binary[int_cols])
    else:
        data_to_steal_binary = convert_label_enc_to_binary(data_to_steal)
        data_to_steal_binary = padding_to_longest_value(data_to_steal_binary)

    column_names = data_to_steal_binary.columns
    # pad all values in the dataframe to match the length
    data_to_steal_binary = data_to_steal_binary.astype(str)

    # THIS IS THE LONGEST VALUE IN THE DATAFRAME (IF ONLY INT VALUES), ALL VALUES PADDED TO THIS LENGTH (IF FLOAT, THEN ALL VALUES ARE 32 BITS)
    len_int_longest_val = float2bin32(int_longest_value)
    binary_string = data_to_steal_binary.apply(lambda x: ''.join(x), axis=1)
    binary_string = ''.join(binary_string.tolist())
    if attack_config.encoding_into_bits == 'direct':
        binary_string = len_int_longest_val + binary_string
        ecc_encoded_data, n_rows_to_hide, n_bits_compressed = ecc_binary_string(n_ecc, binary_string, limit, len(all_columns))
        binary_string = ''.join(format(byte, '08b') for byte in ecc_encoded_data)
        print('string before attack: ', binary_string[:64])

    if attack_config.encoding_into_bits == 'gzip':
        ecc_encoded_data, n_rows_to_hide, n_bits_compressed = compress_binary_string(n_ecc, binary_string, limit, len(all_columns))
        binary_string = ''.join(format(byte, '08b') for byte in ecc_encoded_data)
        n_rows_bits_cap = len(binary_string)

    if attack_config.encoding_into_bits == 'RSCodec':  # attack_config.exfiltration_encoding == 'RSCodec':
        binary_string, n_rows_to_hide, n_rows_bits_cap = (binary_string, limit, len(all_columns))

    return data_to_steal, data_to_steal_binary, binary_string, int_longest_value, longest_value, column_names, cat_cols, int_cols, float_cols, num_cols, num_cat_cols, num_int_cols, num_float_cols, n_rows_to_hide, n_rows_bits_cap, n_bits_compressed



def test_benign_model(X_train, test_dataset, attack_config, model_config, model_path):
    # ========================================
    # BUILD THE MLP
    # AND LOAD THE PARAMS INTO THE MODEL
    # ========================================
    input_size = X_train.shape[1]

    benign_model = build_mlp(input_size, layer_size=model_config.layer_size, num_hidden_layers=model_config.num_hidden_layers, dropout = model_config.dropout)

    # Load the saved model weights from the .pth file
    benign_model.load_state_dict(torch.load(model_path))


    n_lsbs = attack_config.n_lsbs
    # Set the model to evaluation mode
    benign_model.eval()
    params = benign_model.state_dict()
    for name, param in benign_model.named_parameters():
        if param.requires_grad:
            wandb.log({f"Benign model {name}_weights": wandb.Histogram(param.data.cpu().numpy())})

    #NUMBER OF PARAMS IN THE BENIGN MODEL
    num_params = sum(p.numel() for p in params.values())
    # ==========================================================================================


    # ==========================================================================================
    # TEST THE BENIGN MODEL AND LOG THE RESULTS
    # ==========================================================================================
    # TRAINING SET RESULTS

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
               'Benign Model Test Class >50K accuracy': test_class_1_accuracy, 'Benign Model Test set Confusion Matrix Plot': test_cm_plot})
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
    return params_as_bits, params_shape_dict


def calc_capacities(attack_config, binary_string, int_longest_value, longest_value, num_params, num_cat_cols, num_int_cols, num_float_cols, n_rows_to_hide, n_rows_bits_cap):
    #==================================================================================================
    # CALCULATE THE NUMBERS OF BITS TO BE ENCODED GIVEN THE CAPACITY
    #==================================================================================================
    #NUMBER OF BITS THAT CAN BE STOLEN (ENCODED INTO THE MODEL GIVEN THE n_lsbs)
    # the amount of bits that can be encoded into the params based on the chosen number of lsbs
    n_lsbs = attack_config.n_lsbs
    bit_capacity = num_params * n_lsbs

    #NUMBER OF BITS WE WANT TO STEAL
    #num_bits_to_steal should be smaller than bit_capacity
    #assert that the size of the bits to steal can be max equal to bit_capacity (number of parameters * how many bits of each param are used for encoding)

    num_bits_to_steal = len(binary_string)
    limit = int(num_params * n_lsbs)

    #CALCULATE how many rows from the dataframe can be stolen
    #BITS IN ONE ROW (for directly enc.label encoded data - num_cols*32 bits, for one-hot-enc. data numerical columns are converted to binary (32bits long, cat_columns are 1bit per column)))
    if attack_config.dataset == 'adult':
        n_cols = 12
    if attack_config.encoding_into_bits == 'direct':
        if attack_config.exfiltration_encoding == 'one_hot': #attack_config.exfiltration_encoding == 'one_hot':
            n_bits_row = (num_int_cols*int_longest_value) + num_cat_cols + (num_float_cols*32)
            n_dataset_values_capacity = (bit_capacity - 32) / (n_bits_row / n_cols) #number of values that can be hidden in the params
            n_dataset_values_to_hide = (num_bits_to_steal - 32) / (n_bits_row / n_cols)  # number of values from the dataset we want to hide
            n_values_capacity = n_dataset_values_capacity
            n_values_to_hide = n_dataset_values_to_hide + 1
            # number of rows that can be exfiltrated
            n_rows_to_hide = n_dataset_values_to_hide / n_cols  # how many rows of training data we want to steal (num of rows in training data)
            n_rows_capacity = (bit_capacity - 32) / n_bits_row  # how many rows of training data we have the capacity to hide in the model

        if attack_config.exfiltration_encoding == 'label': #attack_config.exfiltration_encoding == 'label':
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
            #print('bin_string', len(binary_string))
            print('The whole training dataset can be exfiltrated')
        elif bit_capacity < len(binary_string):  # if we want to exfiltrate more data than we have capacity for, then we make sure we exfiltrate only up to max capacity
            binary_string = binary_string[:n_rows_bits_cap]

        wandb.log({'Number of LSBs': attack_config.n_lsbs,
                   'Number of parameters in the model': num_params,
                   'Number of parameters used for hiding': n_rows_bits_cap/n_lsbs,
                   'Proportion of parameters used for hiding in %': ((n_rows_bits_cap/n_lsbs)/num_params)*100,
                   'Number of bits to be hidden': num_bits_to_steal,
                   #'Number of bits per data sample': n_bits_row,
                   'Bit capacity (how many bits can be hidden)': bit_capacity,
                   'Number of rows to be hidden': n_rows_to_hide,
                   'Maximum number of rows that can be exfiltrated (capacity)': n_rows_capacity,
                   'Proportion of the dataset stolen in %': min((n_rows_capacity / n_rows_to_hide) * 100, 100)})

    if attack_config.encoding_into_bits == 'gzip' or attack_config.encoding_into_bits == 'RSCodec':
        n_bits_row = n_cols*32
        wandb.log({'Number of LSBs': attack_config.n_lsbs,
                   'Number of parameters in the model': num_params,
                   'Number of parameters used for information hiding': num_bits_to_steal/n_lsbs,
                   'Proportion of parameters used for hiding in %': ((num_bits_to_steal / n_lsbs) / num_params)*100,
                   'Number of bits to be hidden': num_bits_to_steal,
                   'Bit capacity (how many bits can be hidden)': bit_capacity,
                   'Number of rows to be hidden': n_rows_to_hide})


    return n_rows_to_hide, n_rows_bits_cap
    #==================================================================================================

def perform_lsb_attack(attack_config, params_as_bits, binary_string, params_shape_dict, input_size):
    # ==========================================================================================
    # PERFORM THE LSB ENCODING ATTACK
    # ==========================================================================================
    #binary string is the training data we want to hide
    encoded_secret_string = encode_secret(params_as_bits, binary_string, attack_config.n_lsbs)
    modified_params = bits_to_params(encoded_secret_string, params_shape_dict)
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
    malicious_model.load_state_dict(modified_params)

    for name, param in malicious_model.named_parameters():
        try:
            data = param.data.cpu().numpy()
            if not np.isnan(data).any():

                wandb.log({f"Malicious model {name} weights": wandb.Histogram(param.data.cpu().numpy())})
        except ValueError as e:
            print(f"Error occurred for parameter {name}: {e}")
        except Exception as e:
            print(e)
        except IndexError as e:
            print(f"Error occurred for parameter {name}: {e}")


    #print('Testing the model on independent test dataset')
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
    wandb.log({'Attack Test set accuracy': test_acc, 'Attack Test set precision': test_prec, 'Attack Test set recall': test_recall,
              'Attack Test set F1 score': test_f1, 'Attack Test set ROC AUC score': test_roc_auc, 'Attack Test Class <=50K accuracy': test_class_0_accuracy,
               'Attack Test Class >50K accuracy': test_class_1_accuracy, 'Attack Test set Confusion Matrix Plot': test_cm})
    print(f'Test Accuracy: {test_acc}')
    return malicious_model

def test_defended_model(model_config, defended_params, X_train, test_dataset):
    # ==========================================================================================
    # TEST THE EFFECTIVENESS OF THE MALICIOUS MODEL
    # AND LOG THE RESULTS
    # ==========================================================================================
    input_size = X_train.shape[1]
    defended_model = build_mlp(input_size, layer_size=model_config.layer_size, num_hidden_layers=model_config.num_hidden_layers, dropout=model_config.dropout)
    # Load the saved model weights from the .pth file
    defended_model.load_state_dict(defended_params)
    y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(
        defended_model, test_dataset)
    for name, param in defended_model.named_parameters():
        try:
            data = param.data.cpu().numpy()
            if not np.isnan(data).any():

                wandb.log({f"Malicious model {name} weights": wandb.Histogram(param.data.cpu().numpy())})
        except ValueError as e:
            print(f"Error occurred for parameter {name}: {e}")
        except Exception as e:
            print(e)
        except IndexError as e:
            print(f"Error occurred for parameter {name}: {e}")

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
    # Log the model performance metrics after applying defense to WandB
    wandb.log(
        {'Defended Test set accuracy': test_acc, 'Defended Test set precision': test_prec, 'Defended Test set recall': test_recall,
         'Defended Test set F1 score': test_f1, 'Defended Test set ROC AUC score': test_roc_auc,
         'Defended Test Class <=50K accuracy': test_class_0_accuracy,
         'Defended Test Class >50K accuracy': test_class_1_accuracy, 'Defended Test set Confusion Matrix Plot': test_cm_plot})
    return defended_model

def save_modified_model(attack_config, model, defense):
    enc_into_bits = attack_config.encoding_into_bits
    exfiltr_enc = attack_config.exfiltration_encoding
    num_hidden_layers = attack_config.num_hidden_layers
    layer_size = attack_config.layer_size
    n_lsbs = attack_config.n_lsbs
    n_defense_lsbs = attack_config.n_defense_lsbs

    if defense == False:
        model_dir_path = os.path.join(Configuration.MODEL_DIR, attack_config.dataset,'lsb', 'malicious', enc_into_bits, exfiltr_enc)
        model_path = os.path.join(model_dir_path, f'{num_hidden_layers}hl_{layer_size}s_{n_lsbs}attack_LSB_model.pth')  # attack_config.dataset

    if defense == True:
        model_dir_path = os.path.join(Configuration.MODEL_DIR, attack_config.dataset, 'lsb', 'malicious', enc_into_bits, exfiltr_enc)
        model_path = os.path.join(model_dir_path, f'{num_hidden_layers}hl_{layer_size}s_{n_lsbs}attack_{n_defense_lsbs}defense_LSB_model.pth') #attack_config.dataset

    if not os.path.exists(model_dir_path): #attack_config.dataset):
        os.makedirs(os.path.join(model_dir_path))
    torch.save(model.state_dict(), model_path)
    model_name = f'{num_hidden_layers}hl_{layer_size}s_{n_lsbs}attack_{n_defense_lsbs}defense_LSB_model.pth'
    wandb.save(model_name)

    # ==========================================================================================


def reconstruct_data_from_params(attack_config, modified_params, data_to_steal, n_lsbs, n_rows_bits_cap, n_rows_to_hide, column_names, cat_cols, int_cols, float_cols, num_cols, ENC, n_bits_compressed):
    # ==========================================================================================
    # ==========================================================================================
    # RECONSTRUCTION OF SAMPLES FROM LSBs of MODIFIED PARAMS
    # DECODING OF THE SECRET
    # ==========================================================================================
    # ==========================================================================================
    n_ecc = attack_config.n_ecc
    #load modified params
    # Convert model parameters to a binary string
    modified_params_as_bits = params_to_bits(modified_params)
    # Function to extract the least significant x bits

    # Extract the least significant x bits from the binary string
    least_significant_bits = extract_x_least_significant_bits(modified_params_as_bits, n_lsbs, n_rows_bits_cap)
    print("string after attack ", least_significant_bits[:64])

    #print("Least significant {} bits of each parameter:".format(n_lsbs))
    #print(len(least_significant_bits))

    if attack_config.encoding_into_bits == 'direct':
        print("Reconstructing from LSBs")
        if attack_config.exfiltration_encoding == 'label':
            exfiltrated_data, ecc_decoded = reconstruct_from_lsbs(least_significant_bits, column_names, n_rows_to_hide, encoding='label', cat_cols=cat_cols, int_cols = None, num_cols=num_cols, n_ecc=n_ecc)
        if attack_config.exfiltration_encoding == 'one_hot':
            exfiltrated_data, ecc_decoded = reconstruct_from_lsbs(least_significant_bits, column_names, n_rows_to_hide, encoding='one_hot', cat_cols=cat_cols, int_cols=int_cols, num_cols=float_cols, n_ecc=n_ecc)

        if ecc_decoded == True:
            # if we were able to decode the data using ECCs, we calculate similarity
            similarity, num_similarity, cat_similarity = calculate_similarity(data_to_steal, exfiltrated_data, num_cols, cat_cols)
        else:
            # if data cannot be decoded using ECCs, then set similarity to 0 (i.e. attacker cannot reconstruct the data)
            similarity, num_similarity, cat_similarity = 0, 0, 0

    if attack_config.encoding_into_bits == 'gzip':
        exfiltrated_data = reconstruct_gzipped_lsbs(least_significant_bits, ENC, column_names, n_rows_to_hide, n_ecc, n_rows_bits_cap, n_bits_compressed)

        if ecc_decoded == True:
            similarity, num_similarity, cat_similarity = calculate_similarity(data_to_steal, exfiltrated_data, num_cols, cat_cols)
        else:
            similarity, num_similarity, cat_similarity = 0, 0, 0

    elif attack_config.encoding_into_bits == 'RSCodec':
        decoded_decompressed_binary_string = rs_decode_and_decompress(least_significant_bits)
        exfiltrated_data, ecc_decoded = reconstruct_from_lsbs(decoded_decompressed_binary_string, column_names, n_rows_to_hide, encoding='label', cat_cols=cat_cols, int_cols = None, num_cols=num_cols)

        if ecc_decoded == True:
            similarity, num_similarity, cat_similarity = calculate_similarity(data_to_steal, exfiltrated_data, num_cols, cat_cols)
        else:
            similarity, num_similarity, cat_similarity = 0, 0, 0

    return similarity, num_similarity, cat_similarity


def apply_defense(n_defense_lsbs, modified_params, num_params, params_shape_dict):
    # ==========================================================================================
    # PERFORM THE LSB SANITIZATION DEFENSE
    # ==========================================================================================
    # here the binary string contains 0 only to 'erase' hidden information from the model parameters
    limit = n_defense_lsbs*num_params
    params_as_bits = params_to_bits(modified_params)
    binary_string = '0'*limit
    encoded_secret_string = encode_secret(params_as_bits, binary_string, n_defense_lsbs)
    defended_params = bits_to_params(encoded_secret_string, params_shape_dict)
    return defended_params


def run_lsb_attack_eval():
    # Set fixed random number seed
    torch.manual_seed(42)
    torch.set_num_threads(40)
    random_state = 42
    api = wandb.Api()
    wandb.init()
    attack_config = wandb.config


    ENC = 0 #AES.new('1234567812345678'.encode("utf8") * 2, AES.MODE_CBC, 'This is an IV456'.encode("utf8"))

    #Get the model_config so an attack sweep can be run
    # the models on which attack will be implemented are determined by the attack config
    # the model_config for each model (each run of the LSB attack) is then loaded from the respective file
    # the function returns the path to the folder that contains the model and the config file

    model_config, model_path = load_model_config_file(attack_config=attack_config, subset="lsb")

    # Get attack and defense parameters from the config file
    dataset = attack_config.dataset
    exfiltration_encoding = attack_config.exfiltration_encoding
    n_ecc = attack_config.n_ecc
    n_lsbs = attack_config.n_lsbs
    n_defense_lsbs = attack_config.n_defense_lsbs
    wandb.log({"Number of defense LSBs": n_defense_lsbs})

    # Load the preprocessed data
    X_train = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_Xtrain.csv'), index_col=0)
    X_test = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_Xtest.csv'), index_col=0)
    y_test = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_ytest.csv'), index_col=0)
    X_train = X_train.values
    X_test = X_test.values
    y_test = y_test.iloc[:,0].tolist()
    test_dataset = MyDataset(X_test, y_test)


    benign_model, params, num_params, input_size = test_benign_model(X_train, test_dataset, attack_config, model_config, model_path)

    # Calculate the capacity  - the limit for how much data can be encoded
    limit = n_lsbs * num_params

    if n_lsbs < n_defense_lsbs:
        print(f"Skipping run: param_a ({n_lsbs}) < param_b ({n_defense_lsbs})")
        # Finish the wandb run and exit
        wandb.finish()
        sys.exit(0)

    # PREPROCESS THE DATA IN ORDER TO EXFILTRATE IT
    # This is preprocessing of the full dataset the attacker want to steal, so that it can be encoded into the LSBs of the parameter of the model (different than preprocessing for training)
    # Time the preprocessing time
    start_time = time.time()

    data_to_steal, data_to_steal_binary, binary_string, int_longest_value, longest_value, column_names, cat_cols, int_cols, float_cols, num_cols, num_cat_cols, num_int_cols, num_float_cols, n_rows_to_hide_compressed, n_rows_bits_cap, n_bits_compressed = preprocess_data_to_exfiltrate(
        attack_config, limit, ENC)

    end_time = time.time()
    elapsed_time_preprocess = end_time - start_time

    # Logging this in order to be able to aggregate results in the wandb dashboard
    wandb.log({"Aggregated Comparison": 0})

    # Calculate the capacities - number of rows that will be hidden in the model, how many bits will be hidden
    # then we can use this to calculate and log other capacity related metrics

    n_rows_to_hide, n_rows_bits_cap = calc_capacities(attack_config, binary_string, int_longest_value, longest_value, num_params, num_cat_cols, num_int_cols, num_float_cols, n_rows_to_hide_compressed, n_rows_bits_cap)
    if attack_config.encoding_into_bits == 'direct':
        if n_rows_to_hide_compressed < 1:
            n_rows_to_hide = 0
            bits_per_sample = n_bits_compressed
        else:
            n_rows_to_hide = n_rows_to_hide_compressed
            bits_per_sample = n_bits_compressed / n_rows_to_hide
        wandb.log({'Number of rows in the dataset': len(X_train),
                   'Number of rows hidden': n_rows_to_hide,
                   'Number of bits per data sample': bits_per_sample,
                   'Proportion of the dataset hidden in %': min(n_rows_to_hide / (len(X_train)) * 100, 100)})
    if attack_config.encoding_into_bits == 'gzip' or attack_config.encoding_into_bits == 'RSCodec':
        n_rows_to_hide = n_rows_to_hide_compressed
        print('Number of rows in the dataset: ', len(X_train))
        print('Number of rows hidden: ', n_rows_to_hide)
        print('Number of bits per data sample: ', n_rows_bits_cap / n_rows_to_hide)
        print('Proportion of the dataset hidden in %: ', min(n_rows_to_hide / (len(X_train)) * 100, 100))
        wandb.log({'Number of rows in the dataset': len(X_train),
                   'Number of rows hidden': n_rows_to_hide,
                   'Number of bits per data sample': n_bits_compressed/n_rows_to_hide,
                   'Proportion of the dataset hidden in %': min(n_rows_to_hide/(len(X_train)) * 100, 100)})

    # PREPARE PARAMETERS FOR ENCODING
    # get params represented by bits, and the shape dictionary of the parameters
    start_time = time.time()
    params_as_bits, params_shape_dict = prepare_params(params)

    #ENCODE THE PREPROCESSED DATA (BIT REPRESENTATION) INTO THE PREPARED PARAMETERS
    modified_params = perform_lsb_attack(attack_config, params_as_bits, binary_string, params_shape_dict, input_size)
    end_time = time.time()

    #TIME IT TOOK TO MODIFY THE PARAMS OF THE MODEL BY ENCODING THE SECRET
    elapsed_time_modifying_params = end_time - start_time

    # WE TEST THE MODEL WITH PARAMETERS MODIFIED BY THE ATTACK
    malicious_model = test_malicious_model(model_config, modified_params, X_train, test_dataset)

    # SAVE THE MODIFIED, MALICIOUS MODEL ("models/lsb/malicious/...)
    # FILENAME: {num_hidden_layers}hl_{layer_size}s_{n_lsbs}attack_LSB_model.pth'
    save_modified_model(attack_config, malicious_model, defense=False)

    # RECONSTRUCT THE DATA FROM THE PARAMETERS AND CALCULATE SIMILARITY TO ORIGINAL DATA
    try:
        print("trying to reconstruct: attack")
        similarity, num_similarity, cat_similarity = reconstruct_data_from_params(attack_config, modified_params, data_to_steal, n_lsbs, n_rows_bits_cap,
                                                  n_rows_to_hide, column_names, cat_cols, int_cols, float_cols, num_cols,
                                                  ENC, n_bits_compressed)
    # UNLESS A DEFENSE IS APPLIED, RECONSTRUCTION WILL ALWAYS RESULT IN 100% SIMILARITY
    except Exception as recon_failed:
        similarity, num_similarity, cat_similarity = 1.00, 1.00, 1.00
    else:
        # Log the success message or any other information to W&B
        wandb.log({"Attack Reconstruction": 'Successful'})

    wandb.log({"Data similarity: Attack": similarity})
    wandb.log({"Numerical Columns Similarity: Attack": num_similarity})
    wandb.log({"Categorical Columns Similarity: Attack": cat_similarity})

    # Record the reconstruction time
    elapsed_time_reconstruction = end_time - start_time

    # Record the attack time - extra preprocessing for attack + time to modify parameters of the model
    elapsed_time = elapsed_time_preprocess + elapsed_time_modifying_params

    # APPLY LSB SANITIZATION DEFENSE AND GET THE PARAMS AFTER THEY WERE MODIFIED BY DEFENSE
    defended_params = apply_defense(n_defense_lsbs, modified_params, num_params, params_shape_dict)

    # GET THE DEFENDED MODEL AND TEST AND LOG ITS PERFORMANCE
    defended_model = test_defended_model(model_config, defended_params, X_train, test_dataset)

    #SAVE THE DEFENDED MODEL  ("models/lsb/malicious/...)
    # FILENAME: {num_hidden_layers}hl_{layer_size}s_{n_lsbs}attack_{n_defense_lsbs}defense_LSB_model.pth'
    save_modified_model(attack_config, defended_model, defense=True)


    try:
        # TRY RECONSTRUCTING THE SECRET FROM THE PARAMETERS MODIFIED BY THE DEFENSE AND CALCULATE SIMILARITY
        print("trying to reconstruct: defense")
        similarity, num_similarity, cat_similarity = reconstruct_data_from_params(attack_config, defended_params, data_to_steal, n_lsbs,
                                                  n_rows_bits_cap, n_rows_to_hide, column_names, cat_cols, int_cols,
                                                  float_cols, num_cols, ENC, n_bits_compressed)

    except Exception as recon_failed:
        # RECONSTRUCTION OF THE SECRET FROM THE PARAMETERS MODIFIED BY THE DEFENSE FAILED - DEFENSE WAS EFFECTIVE
        # Attacker did not reconstruct the data at all, i.e. similarity is 0
        similarity, num_similarity, cat_similarity = 0, 0, 0
        wandb.log({"Reconstruction after defense": 'Failed'})
    else:
        # IF the attacker reconstructed, log successful reconstruction (check similarity to see how effective the defense was)
        wandb.log({"Reconstruction after defense": 'Successful'})

    wandb.log({"Data Similarity: Defense": similarity})
    wandb.log({"Numerical Columns Similarity: Defense": num_similarity})
    wandb.log({"Categorical Columns Similarity: Defense": cat_similarity})
    wandb.log({'LSB Attack Time': elapsed_time})
    wandb.log({'LSB Reconstruction Time': elapsed_time_reconstruction})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    run_lsb_attack_eval()
