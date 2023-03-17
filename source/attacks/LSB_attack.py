import os.path
import types

import numpy as np
import torch
import wandb
import yaml
from sklearn.model_selection import train_test_split

from source.data_loading.data_loading import get_X_y_for_network, MyDataset
from source.evaluation.evaluation import eval_on_test_set
from lsb_helpers import params_to_bits, bits_to_params, float2bin32, convert_label_enc_to_binary, \
    convert_one_hot_enc_to_binary, encode_secret
from source.networks.network import build_network, build_mlp
from source.utils.Configuration import Configuration
from source.utils.wandb_helpers import load_model_config_file

api = wandb.Api()
project = "Data_Exfiltration_Attacks_and_Defenses"
wandb.init(project=project)
attack_config = wandb.config

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

#TODO calculate confidence intervals
#TODO check how to load config values from different runs/sweeps
#TODO encode data as categorical to save space
#TODO calculate how many bits are saved by the one hot encoding

# Set fixed random number seed
torch.manual_seed(42)
torch.set_num_threads(32)


#get the model_config so an attack sweep can be run
# the models on which attack will be implemented are determined by the attack config
# the model_config for each model (each run of the LSB attack) is then loaded from the respective file
model_config = load_model_config_file(attack_config=attack_config)


# ==========================
# === DATA FOR TRAINING ===
# ==========================
X_train, y_train, X_test, y_test, encoders = get_X_y_for_network(model_config, purpose='train')
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
input_size = X_train.shape[1]
train_dataset = MyDataset(X_train, y_train)
val_dataset = MyDataset(X_val, y_val)
test_dataset = MyDataset(X_test, y_test)

# ===========================
# == DATA FOR EXFILTRATION ==
# ===========================
X_train_ex, y_train_ex, X_test_ex, y_test_ex, encoders = get_X_y_for_network(model_config, purpose='exfiltrate', exfiltration_encoding=attack_config.exfiltration_encoding)
numerical_columns = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
X_train_ex['income'] = y_train_ex
data_to_steal = X_train_ex

# NUMBER OF DATA TO EXFILTRATE
num_values_df = data_to_steal.size

# ========================================
# DATA TO EXILTRATE WILL BE LABEL ENCODED
# AND DIRECTLY CONVERTED TO BITS
# ========================================
if attack_config.exfiltration_encoding == 'label':
    data_to_steal_binary = convert_label_enc_to_binary(data_to_steal)
# ========================================
# DATA TO EXILTRATE WILL BE ONE-HOT ENCODED
# AND DIRECTLY CONVERTED TO BITS
# ========================================
# ONE HOT ENCODED DATA WILL BE ENCODED DIRECTLY INTO PARAMETERS IN ORDER TO SAVE SPACE
# 41 COLUMNS FOR ADULT DATASET, DOES NOT MAKE SENSE TO COMPRESS, BECAUSE LABEL ENCODED DATA CAN BE COMPRESSED MROE - 12 COLUMNS ONLY
if attack_config.exfiltration_encoding == 'one_hot':
    data_to_steal_binary = convert_one_hot_enc_to_binary(data_to_steal, numerical_columns)

#pad all values in the dataframe to match the length
for col in data_to_steal_binary:
    data_to_steal_binary[col] = data_to_steal_binary[col].apply(lambda x: x.zfill(32))
binary_string = data_to_steal_binary.apply(lambda x: ''.join(x), axis=1)
binary_string = ''.join(binary_string.tolist())

print('length of bits to steal: ', len(binary_string))

# ========================================
# BUILD THE MLP
# AND LOAD THE PARAMS INTO THE MODEL
# ========================================
model = build_mlp(input_size, layer_size=model_config.layer_size, num_hidden_layers=model_config.num_hidden_layers, batch_size= model_config.batch_size, dropout = model_config.dropout)
# Load the saved model weights from the .pth file
model.load_state_dict(torch.load('models/adult/benign/best_benign_adult.pth'))
wandb.watch(model, log='all')

n_lsbs = attack_config.n_lsbs
# Set the model to evaluation mode
model.eval()
params = model.state_dict()
# ==========================================================================================




# ==========================================================================================
# TEST THE BENIGN MODEL AND LOG THE RESULTS
# ==========================================================================================
#test_dataset = MyDataset(X_test, y_test)
print('Testing the model on independent test dataset')
y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(model, test_dataset, model_config.threshold)
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
# Log the training and validation metrics to WandB
wandb.log({'Test set accuracy': test_acc, 'Test set precision': test_prec, 'Test set recall': test_recall,
           'Test set F1 score': test_f1, 'Test set ROC AUC score': test_roc_auc,
           'Test Class <=50K accuracy': test_class_0_accuracy,
           'Test Class >50K accuracy': test_class_1_accuracy, 'Test set Confusion Matrix Plot': test_cm})
# cm for train and val build with predictions averaged over all folds
print(f'Test Accuracy: {test_acc}')
# ==========================================================================================



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



#==================================================================================================
# CALCULATE THE NUMBERS OF BITS TO BE ENCODED GIVEN THE CAPACITY
#==================================================================================================
#NUMBER OF PARAMS IN THE BENIGN MODEL
num_params = sum(p.numel() for p in params.values())
#NUMBER OF BITS THAT CAN BE STOLEN (ENCODED INTO THE MODEL GIVEN THE n_lsbs)
bit_capacity = num_params * n_lsbs # the amount of bits that can be encoded into the params based on the chosen number of lsbs
#NUMBER OF BITS WE WANT TO STEAL
num_bits_to_steal = len(binary_string) #num_bits_to_steal should be smaller than bit_capacity
#assert that the size of the bits to steal can be max equal to bit_capacity (number of parameters * how many bits of each param are used for encoding)

#CALCULATE how many rows from the dataframe can be stolen
#BITS IN ONE ROW (for directly enc.label encoded data - num_cols*32 bits, for one-hot-enc. data numerical columns are converted to binary (32bits long, cat_columns are 1bit per column)))
# Get the list of all columns in the dataframe
all_cols = data_to_steal.columns.tolist()
# Get the list of categorical columns by removing the numerical columns
cat_cols = [col for col in all_cols if col not in numerical_columns]
# Count the number of numerical and categorical columns
num_col_count = len(numerical_columns)
cat_col_count = len(cat_cols)
if attack_config.encoding_into_bits == 'direct':
    n_cols = data_to_steal.shape[1]
    if attack_config.exfiltration_encoding == 'one_hot':
        n_bits_row = (num_col_count*32) + cat_col_count
    if attack_config.exfiltration_encoding == 'label':
        n_bits_row = n_cols * 32

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

wandb.log({'Number of LSBs': attack_config.n_lsbs, 'Number of parameters in the model':num_params, '# of bits to be hidden': num_bits_to_steal, 'Bit capacity (how many bits can be hidden)': bit_capacity, 'Number of rows to be hidden':n_rows_to_hide, 'Maximum # of rows that can be exfiltrated (capacity)': n_rows_capacity, 'Proportion of the dataset stolen': (bit_capacity/num_bits_to_steal)*100})
#==================================================================================================


# ==========================================================================================
# PERFORM THE LSB ENCODING ATTACK
# ==========================================================================================
#binary string is the training data we want to hide
encoded_secret_string = encode_secret(params_as_bits, binary_string, attack_config.n_lsbs)
modified_params = bits_to_params(encoded_secret_string, params_shape_dict)
malicious_model = build_mlp(input_size, model_config.layer_size, model_config.num_hidden_layers, model_config.dropout)
malicious_model.load_state_dict(modified_params)
# ==========================================================================================


# ==========================================================================================
# TEST THE EFFECTIVENESS OF THE MALICIOUS MODEL
# AND LOG THE RESULTS
# ==========================================================================================
#test_dataset = MyDataset(X_test, y_test)
print('Testing the model on independent test dataset')
y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(malicious_model, test_dataset, model_config.threshold)
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
mal_model_path = os.path.join(Configuration.MODEL_DIR, attack_config.dataset, 'malicious', f'{n_lsbs}_LSB_model.pth')
torch.save(malicious_model.state_dict(), mal_model_path)
wandb.save('LSB_model.pth')
# ==========================================================================================




# ==========================================================================================
# ==========================================================================================

# RECONSTRUCTION OF SAMPLES FROM LSBs of MODIFIED PARAMS

# ==========================================================================================
# ==========================================================================================

#load modified params
# Convert model parameters to a binary string
modified_params_as_bits = params_to_bits(modified_params)
# Function to extract the least significant x bits
def extract_x_least_significant_bits(binary_str, n_lsbs):
    step = 32
    extracted_bits = ""
    for i in range(0, len(binary_str), step):
        extracted_bits += binary_str[i - n_lsbs:i]
    return extracted_bits

# Extract the least significant x bits from the binary string
least_significant_bits = extract_x_least_significant_bits(modified_params_as_bits, n_lsbs)
print("Least significant {} bits of each parameter:".format(n_lsbs))
print(least_significant_bits)

#TODO turn extracted bits into the shape of data_to_steal (based on if they were one-hot or label encoded)
# convert the binary strings into float values (and round them up or down)
# compare to original data
# define similarity function
# reconstruction if compression and encryption done
# implement defense by flipping LSBs, measure the impact on effectiveness and on similarity of exfiltrated data vs original data
# measure the impact of the defense when applied preventatively on a benign model

#TODO
# a model user/client will want the smallest model at the tradeoff for accuracy, so not a lot of data can be encoded in there
# COMPARE: when defense applied on a smaller model - does it have a bigger impact than on a bigger model? then maybe the defender prefers bigger model with the defense, than small model
# ADAPTIVE ATTACKER if the defender requires a small model - compression used fit more data into the model - can it be exfiltrated even though some defense is applied ?
