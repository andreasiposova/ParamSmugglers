import os

import pandas as pd
import torch
import numpy as np
import wandb

import argparse
import os
import numpy as np
import pandas as pd

import copy
import torch
import wandb
import math
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from wandb.wandb_torch import torch as wandb_torch

from torch.autograd import Variable
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

import matplotlib.pyplot as plt

from source.attacks.SE_helpers import reconstruct_from_signs
from source.attacks.black_box_helpers import generate_malicious_data, reconstruct_from_preds, cm_class_acc, baseline
from source.attacks.lsb_helpers import convert_label_enc_to_binary
from source.attacks.similarity import calculate_similarity
from source.data_loading.data_loading import MyDataset
from source.evaluation.evaluation import eval_on_test_set
from source.networks.network import MLP_Net, MLP_Net_x, build_mlp, build_optimizer


from source.utils.Configuration import Configuration



def modify_signs(model, percentage_to_modify):
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
                    #TODO return indices of the modified values
    return model




def eval_defense(config, X_train, y_train, X_test, y_test, X_triggers, y_triggers, column_names, n_rows_to_hide, data_to_steal, hidden_num_cols, hidden_cat_cols, mal_ratio, repetition, mal_data_generation):
    wandb.init()
    # Parameters
    #batch_size = 512
    #class_weights = 'not_applied'
    #dataset = 'adult'
    #dropout = 0
    #encoding = 'one_hot'
    #epochs = 120
    #layer_size = 5
    #learning_rate = 0.001
    #mal_data_generation = 'uniform'
    #mal_ratio = 0.1
    #num_hidden_layers = 3
    #optimizer_name = 'adam'
    #ratio = 'equal'
    #repetition = 4
    #weight_decay = 0
    #pruning_amount = 0.02
    dataset = config.dataset
    mal_ratio = config.mal_ratio
    mal_data_generation = config.mal_data_generation
    repetition = config.repetition
    layer_size = config.layer_size
    num_hidden_layers = config.num_hidden_layers
    #pruning_amount = config.pruning_amount
    dropout = config.dropout
    # Input size
    input_size = 41
    pruning_amount_config = config.pruning_amount

    number_of_samples = len(X_train)
    number_of_samples2gen = int(number_of_samples * mal_ratio)
    num_of_cols = len(data_to_steal.columns)
    bits_per_row = num_of_cols * 32
    n_rows_to_hide = int(math.floor(number_of_samples2gen / bits_per_row))

    X_triggers_rep = X_triggers.copy()
    for _ in range(repetition - 1):
        X_triggers_rep = np.append(X_triggers_rep, X_triggers, axis=0)
    y_triggers_rep = y_triggers*repetition

    train_dataset = MyDataset(X_train, y_train)
    #val_dataset = MyDataset(X_val_cv, y_val_cv)  # 20% of the training data
    test_dataset = MyDataset(X_test, y_test)  # separate test set
    trigger_dataset = MyDataset(X_triggers_rep, y_triggers_rep)
    X_train = train_dataset.X
    y_train = train_dataset.y
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)

    X_test = test_dataset.X
    y_test = test_dataset.y
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32)

    #if not os.path.exists(os.path.join(Configuration.MODEL_DIR, f'{dataset}/black_box/benign/{num_hidden_layers}hl_{layer_size}s')):
    #    os.makedirs(os.path.join(Configuration.MODEL_DIR, f'{dataset}/black_box/benign/{num_hidden_layers}hl_{layer_size}s'))
    #if not os.path.exists(os.path.join(Configuration.MODEL_DIR, f'{dataset}/black_box/malicious/{num_hidden_layers}hl_{layer_size}s')):
    #    os.makedirs(os.path.join(Configuration.MODEL_DIR, f'{dataset}/black_box/malicious/{num_hidden_layers}hl_{layer_size}s'))
    ben_state_dict = torch.load(os.path.join(Configuration.MODEL_DIR, f'{dataset}/black_box/benign/{num_hidden_layers}hl_{layer_size}s/{mal_ratio}ratio_{repetition}rep_{mal_data_generation}.pth'))
    mal_state_dict = torch.load(os.path.join(Configuration.MODEL_DIR, f'{dataset}/black_box/malicious/{num_hidden_layers}hl_{layer_size}s/{mal_ratio}ratio_{repetition}rep_{mal_data_generation}.pth'))

    # Build the benign and attacked MLP model
    benign_model = build_mlp(input_size, layer_size, num_hidden_layers, dropout)
    attacked_model = build_mlp(input_size, layer_size, num_hidden_layers, dropout)

    params = benign_model.state_dict()
    num_params = sum(p.numel() for p in params.values())
    #Build the optimizer
    #optimizer = build_optimizer(benign_model, optimizer_name, learning_rate, weight_decay)

    # Load the state dict into the models
    benign_model.load_state_dict(ben_state_dict)
    attacked_model.load_state_dict(mal_state_dict)



    percent_to_modify = int(0.02*100)

    modification_range = list(range(0, 100, percent_to_modify))
    print(modification_range)

    benign_output = benign_model.forward(X_train)
    attacked_output = attacked_model.forward(X_train)

    # Convert the output to binary labels
    benign_output_labels = (benign_output > 0.5).float().detach().numpy()
    attacked_output_labels = (attacked_output > 0.5).float().detach().numpy()

    # Detach the output tensor for the roc_auc_score function
    benign_output_np = benign_output.detach().numpy()
    attacked_output_np = attacked_output.detach().numpy()

    # Assuming y_train is a numpy array. If y_train is a tensor, it should also be detached before using it
    y_train_np = y_train.detach().numpy() if isinstance(y_train, torch.Tensor) else y_train

    _, benign_model_activations = benign_model.forward_act(X_train)
    _, attacked_model_activations = attacked_model.forward_act(X_train)
    pr_step = 0

    for step in modification_range:

        # Then, we pass the X_train data through the network to get the output and activations
        #TODO keep the indices of the changed signs, to change different signs in each step
        if step > 0:
            ben_model = modify_signs(benign_model, step)
            att_model, attacked_pruned_indices = modify_signs(attacked_model, step)

        if step == 0:
            ben_model, benign_pruned_indices = modify_signs(benign_model, step)
            att_model, attacked_pruned_indices = modify_signs(attacked_model, step)
            exfiltrated_data = reconstruct_from_signs(att_model, column_names, n_rows_to_hide)
            similarity = calculate_similarity(data_to_steal, exfiltrated_data, hidden_num_cols, hidden_cat_cols)
            start_similarity = similarity
        else:

            ben_model, benign_pruned_indices = modify_signs(ben_model, step)
            att_model, attacked_pruned_indices = modify_signs(att_model, step)



        y_trigger_ints, y_trigger_preds_ints, trigger_acc, trigger_prec, trigger_recall, trigger_f1, trigger_roc_auc, trigger_cm = eval_on_test_set(att_model, trigger_dataset)
        exfiltrated_data = reconstruct_from_preds(y_trigger_preds_ints, column_names, n_rows_to_hide)
        similarity = calculate_similarity(data_to_steal, exfiltrated_data, hidden_num_cols, hidden_cat_cols)
        similarity = similarity/100


        benign_output = ben_model.forward(X_train)
        attacked_output = att_model.forward(X_train)
        # Convert the output to binary labels
        benign_output_labels = (benign_output > 0.5).float().detach().numpy()
        attacked_output_labels = (attacked_output > 0.5).float().detach().numpy()

        # Detach the output tensor for the roc_auc_score function
        benign_output_np = benign_output.detach().numpy()
        attacked_output_np = attacked_output.detach().numpy()

        # Assuming y_train is a numpy array. If y_train is a tensor, it should also be detached before using it
        y_train_np = y_train.detach().numpy() if isinstance(y_train, torch.Tensor) else y_train

        # Compute the evaluation metrics
        base_train_acc_e = accuracy_score(y_train_np, benign_output_labels)
        base_train_prec_e = precision_score(y_train_np, benign_output_labels)
        base_train_recall_e = recall_score(y_train_np, benign_output_labels)
        base_train_f1_e = f1_score(y_train_np, benign_output_labels)
        base_train_roc_auc_e = roc_auc_score(y_train_np, benign_output_np)
        train_acc_e = accuracy_score(y_train_np, attacked_output_labels)
        train_prec_e = precision_score(y_train_np, attacked_output_labels)
        train_recall_e = recall_score(y_train_np, attacked_output_labels)
        train_f1_e = f1_score(y_train_np, attacked_output_labels)
        train_roc_auc_e = roc_auc_score(y_train_np, attacked_output_np)



        att_y_test_ints, att_y_test_preds_ints, att_test_acc, att_test_prec, att_test_recall, att_test_f1, att_test_roc_auc, att_test_cm = eval_on_test_set(att_model, test_dataset)
        base_y_test_ints, base_y_test_preds_ints, base_test_acc, base_test_prec, base_test_recall, base_test_f1, base_test_roc_auc, base_test_cm = eval_on_test_set(ben_model, test_dataset)



def run_sm_defense():
    api = wandb.Api()
    project = "Sign_Encoding"
    wandb.init(project=project)

    seed = 42
    np.random.seed(seed)
    #config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'Black_box_adult_sweep')
    #attack_config = load_config_file(config_path)
    attack_config = wandb.config
    dataset = attack_config.dataset
    mal_ratio = attack_config.mal_ratio
    mal_data_generation = attack_config.mal_data_generation
    repetition = attack_config.repetition
    layer_size = attack_config.layer_size
    num_hidden_layers = attack_config.num_hidden_layers
    pruning_amount = attack_config.pruning_amount

    #batch_size = 512
    #class_weights = 'not_applied'
    #dataset = 'adult'
    #dropout = 0
    #encoding = 'one_hot'
    #epochs = 120
    #layer_size = 5
    #learning_rate = 0.001
    #mal_data_generation = 'uniform'
    #mal_ratio = 0.1
    #num_hidden_layers = 3
    #optimizer_name = 'adam'
    #ratio = 'equal'
    #repetition = 4
    #weight_decay = 0
    #pruning_amount = 0.05

    # Input size
    input_size = 41


    if dataset == 'adult':
        X_train = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_to_steal_one_hot.csv'), index_col=0)
        X_train = X_train.iloc[:,:-1]
        X_test = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_Xtest.csv'), index_col=0)
        y_test = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_ytest.csv'), index_col=0)
        y_train = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_ytrain.csv'), index_col=0)
        y_test = y_test.iloc[:,0].tolist()
        y_train = y_train.iloc[:, 0].tolist()

        all_column_names = X_train.columns
        data_to_steal = pd.read_csv(os.path.join(Configuration.TAB_DATA_DIR, f'{dataset}_data_to_steal_label.csv'),index_col=0)
        hidden_col_names = data_to_steal.columns
        hidden_num_cols = ["age", "education_num", "capital_change", "hours_per_week"]
        hidden_cat_cols = [col for col in hidden_col_names if col not in hidden_num_cols]