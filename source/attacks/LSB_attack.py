import os.path
import types

import numpy as np
import torch
import wandb
import yaml
from sklearn.model_selection import train_test_split

from source.data_loading.data_loading import get_X_y_for_network, MyDataset
from source.evaluation.evaluation import eval_on_test_set
from lsb_helpers import params_to_bits, bits_to_params, float2bin32
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

# ==========================
# == DATA FOR EXFILTRATION =
# ==========================
X_train_ex, y_train_ex, X_test_ex, y_test_ex, encoders = get_X_y_for_network(model_config, purpose='exfiltrate', exfiltration_encoding=attack_config.exfiltration_encoding)
numerical_columns = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
X_train_ex['income'] = y_train_ex
data_to_steal = X_train_ex

# NUMBER OF DATA FOR TRAINING
num_values_df = data_to_steal.size

if attack_config.exfiltration_encoding == 'label':
    for col in data_to_steal:
        data_to_steal[col] = data_to_steal[col].astype(np.int32)
if attack_config.exfiltration_encoding == 'one_hot':
    for col in numerical_columns:
        data_to_steal[col] = data_to_steal[col].astype(np.int32)

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


model = build_mlp(input_size, layer_size=model_config.layer_size, num_hidden_layers=model_config.num_hidden_layers, batch_size= model_config.batch_size, dropout = model_config.dropout)
# Load the saved model weights from the .pth file
model.load_state_dict(torch.load('models/adult/benign/1hl_3d_best_benign_adult.pth'))
wandb.watch(model, log='all')

n_lsbs = attack_config.n_lsbs
# Set the model to evaluation mode
model.eval()
params = model.state_dict()

#test_dataset = MyDataset(X_test, y_test)
print('Testing the model on independent test dataset')
y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(
    model, test_dataset, model_config.threshold)

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

wandb.log({'Number of LSBs': attack_config.n_lsbs, 'Number of parameters in the model':num_params, '# of bits to be hidden': num_bits_to_steal, 'Bit capacity (how many bits can be hidden)': bit_capacity, 'Number of rows to be hidden':n_rows_to_hide, 'Maximum # of rows that can be exfiltrated (capacity)': n_rows_capacity, 'Proportion of the dataset stolen': (bit_capacity/num_bits_to_steal)*100})

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
malicious_model = build_mlp(input_size, model_config.layer_size, model_config.num_hidden_layers, model_config.dropout)
malicious_model.load_state_dict(modified_params)

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
torch.save(malicious_model.state_dict(), 'LSB_model.pth')
wandb.save('LSB_model.pth')
wandb.finish()
# Print the names and shapes of each parameter tensor



#load modified params
# Convert model parameters to a binary string
modified_params_as_bits = params_to_bits(modified_params)
# Function to extract the least significant x bits
def extract_x_least_significant_bits(binary_str, x):
    step = 32
    extracted_bits = ""
    for i in range(x, len(binary_str), step):
        extracted_bits += binary_str[i - x:i]
    return extracted_bits

# Extract the least significant x bits from the binary string
n_lsbs = 4
least_significant_bits = extract_x_least_significant_bits(modified_params_as_bits, n_lsbs)
print("Least significant {} bits of each parameter:".format(n_lsbs))
print(least_significant_bits)