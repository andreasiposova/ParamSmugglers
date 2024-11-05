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

"""

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
"""

import torch
import numpy as np

class WeightModifier:
    def __init__(self, model):
        self.model = model
        self.previously_modified_indices = {name: torch.tensor([], dtype=torch.long)
                                            for name, param in model.named_parameters()
                                            if 'weight' in name}

    def modify_signs(self, percentage_to_modify):
        modified_indices = {}

        for name, param in self.model.named_parameters():
            if 'weight' in name:
                with torch.no_grad():
                    flat_weights = param.view(-1)
                    total_weights = flat_weights.size(0)
                    n_modify = int(percentage_to_modify * total_weights)

                    # Exclude previously modified indices
                    available_indices = torch.ones(total_weights, dtype=torch.bool)
                    available_indices[self.previously_modified_indices[name]] = False
                    available_indices = torch.where(available_indices)[0]

                    if available_indices.size(0) < n_modify:
                        #raise ValueError("Not enough unmodified weights left to modify. Consider resetting.")
                        break

                    # Select new weights to modify
                    _, new_indices = torch.topk(flat_weights[available_indices].abs(), n_modify, largest=False)
                    actual_indices_to_modify = available_indices[new_indices]

                    # Determine range for new values
                    n_range_determination = int(0.1 * total_weights)
                    _, range_indices = torch.topk(flat_weights.abs(), n_range_determination, largest=False)
                    max_val_in_range = flat_weights[range_indices].max()

                    # Modify weights
                    for idx in actual_indices_to_modify:
                        new_value = np.random.uniform(0, max_val_in_range.item())
                        flat_weights[idx] = -new_value if flat_weights[idx] > 0 else new_value

                    # Store and update tracking of modified indices
                    modified_indices[name] = actual_indices_to_modify.tolist()
                    if self.previously_modified_indices[name].numel() == 0:
                        self.previously_modified_indices[name] = actual_indices_to_modify
                    else:
                        self.previously_modified_indices[name] = torch.cat([self.previously_modified_indices[name], actual_indices_to_modify])

        return self.model, modified_indices

    def reset_modified_indices(self):
        for key in self.previously_modified_indices:
            self.previously_modified_indices[key] = torch.tensor([], dtype=torch.long)





def eval_defense(config, X_train, y_train, X_test, y_test, column_names, data_to_steal, hidden_num_cols, hidden_cat_cols):
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
    layer_size = config.layer_size
    num_hidden_layers = config.num_hidden_layers
    dropout = config.dropout
    lambda_s = config.lambda_s
    percent_to_modify = config.percent_to_modify
    # Input size
    if dataset == 'adult':
        input_size = 41

    number_of_samples = len(X_train)
    num_of_cols = len(data_to_steal.columns)
    bits_per_row = num_of_cols * 32

    train_dataset = MyDataset(X_train, y_train)
    #val_dataset = MyDataset(X_val_cv, y_val_cv)  # 20% of the training data
    test_dataset = MyDataset(X_test, y_test)  # separate test set
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
    ben_state_dict = torch.load(os.path.join(Configuration.MODEL_DIR, f'{dataset}/sign_encoding/benign/{num_hidden_layers}hl_{layer_size}s/penalty_{lambda_s}.pth'))
    mal_state_dict = torch.load(os.path.join(Configuration.MODEL_DIR, f'{dataset}/sign_encoding/malicious/{num_hidden_layers}hl_{layer_size}s/penalty_{lambda_s}.pth'))

    # Build the benign and attacked MLP model
    benign_model = build_mlp(input_size, layer_size, num_hidden_layers, dropout)
    attacked_model = build_mlp(input_size, layer_size, num_hidden_layers, dropout)

    params = benign_model.state_dict()
    num_params = sum(p.numel() for p in params.values())
    #Build the optimizer
    #optimizer = build_optimizer(benign_model, optimizer_name, learning_rate, weight_decay)
    n_rows_to_hide = int(math.floor(num_params / bits_per_row))

    # Load the state dict into the models
    benign_model.load_state_dict(ben_state_dict)
    attacked_model.load_state_dict(mal_state_dict)

    percentage_to_modify = int(percent_to_modify*100)
    modification_range = list(range(0, 100, percentage_to_modify))
    print(modification_range)
    base_modifier = WeightModifier(benign_model)
    mal_modifier = WeightModifier(attacked_model)

    # Assuming y_train is a numpy array. If y_train is a tensor, it should also be detached before using it
    y_train_np = y_train.detach().numpy() if isinstance(y_train, torch.Tensor) else y_train


    for step in modification_range:
        print(step)

        # Then, we pass the X_train data through the network to get the output and activations
        #TODO keep the indices of the changed signs, to change different signs in each step
        #if step > 0:
        ben_model, modified_base_indices = base_modifier.modify_signs(percent_to_modify)
        att_model, modified_attack_indices = mal_modifier.modify_signs(percent_to_modify)

        #if step == 0:
         #   ben_model, modified_base_indices = base_modifier.modify_signs(percent_to_modify)
          #  att_model, modified_attack_indices = mal_modifier.modify_signs(percent_to_modify)
            #exfiltrated_data = reconstruct_from_signs(att_model, column_names, n_rows_to_hide)
            #similarity = calculate_similarity(data_to_steal, exfiltrated_data, hidden_num_cols, hidden_cat_cols)
            #start_similarity = similarity

        #else:
         #   ben_model, modified_base_indices = base_modifier.modify_signs(ben_model, percent_to_modify)
          #  att_model, modified_attack_indices = mal_modifier.modify_signs(att_model, percent_to_modify)




        exfiltrated_data = reconstruct_from_signs(att_model, column_names, n_rows_to_hide)
        similarity = calculate_similarity(data_to_steal, exfiltrated_data, hidden_num_cols, hidden_cat_cols)
        similarity = similarity/100
        print(similarity)


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

        # RESULTS OF THE BASE NETWORK ON THE BENIGN TRAIN DATA
        base_train_class_0_accuracy, base_train_class_1_accuracy, base_benign_train_cm, base_train_tn, base_train_fp, base_train_fn, base_train_tp = cm_class_acc(benign_output_labels, y_train_np)
        # RESULTS OF THE BASE NETWORK ON THE TEST DATA
        base_test_class_0_accuracy, base_test_class_1_accuracy, base_test_cm, base_test_tn, base_test_fp, base_test_fn, base_test_tp = cm_class_acc(base_y_test_preds_ints, base_y_test_ints)

        # RESULTS OF THE MALICIOUS NETWORK ON THE BENIGN TRAIN DATA
        mal_benign_train_class_0_accuracy, mal_benign_train_class_1_accuracy, mal_benign_train_cm, mal_train_tn, mal_train_fp, mal_train_fn, mal_train_tp = cm_class_acc(attacked_output_labels, y_train_np)

        # RESULTS OF THE MALICIOUS NETWORK ON THE TEST DATA
        mal_test_class_0_accuracy, mal_test_class_1_accuracy, mal_test_cm, mal_test_tn, mal_test_fp, mal_test_fn, mal_test_tp = cm_class_acc(att_y_test_preds_ints, att_y_test_ints)





        mal_train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_train_np, preds=attacked_output_labels, class_names=["<=50K", ">50K"])
        mal_test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=att_y_test_ints, preds=att_y_test_preds_ints, class_names=["<=50K", ">50K"])

        base_train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_train_np,preds=benign_output_labels, class_names=["<=50K", ">50K"])
        base_test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=base_y_test_ints, preds=base_y_test_preds_ints, class_names=["<=50K", ">50K"])


        baseline_test = baseline(base_y_test_ints.tolist())
        baseline_train = baseline(y_train_np.tolist())

        results = {'Malicious Model: Training set accuracy': train_acc_e,
                   'Malicious Model: Training set precision': train_prec_e,
                   'Malicious Model: Training set recall': train_recall_e,
                   'Malicious Model: Training set F1 score': train_f1_e,
                   'Malicious Model: Training set ROC AUC score': train_roc_auc_e,
                   'Malicious Model: Test set accuracy': att_test_acc,
                   'Malicious Model: Test set precision': att_test_prec,
                   'Malicious Model: Test set recall': att_test_recall,
                   'Malicious Model: Test set F1 score': att_test_f1,
                   'Malicious Model: Test set ROC AUC score': att_test_roc_auc,
                   'Malicious Model: Training set Class 1 Accuracy': mal_benign_train_class_1_accuracy,
                   'Malicious Model: Training set Class 0 Accuracy': mal_benign_train_class_0_accuracy,
                   'Malicious Model: Test Set Class 1 Accuracy': mal_test_class_1_accuracy,
                   'Malicious Model: Test Set Class 0 Accuracy': mal_test_class_0_accuracy,
                   'Similarity after step': similarity,
                   'Base Model: Training set accuracy': base_train_acc_e,
                   'Base Model: Training set precision': base_train_prec_e,
                   'Base Model: Training set recall': base_train_recall_e,
                   'Base Model: Training set F1 Score': base_train_f1_e,
                   'Base Model: Training set ROC AUC': base_train_roc_auc_e,
                   'Base Model: Test set accuracy': base_test_acc,
                   'Base Model: Test set precision': base_test_prec,
                   'Base Model: Test set recall': base_test_recall,
                   'Base Model: Test set F1 Score': base_test_f1,
                   'Base Model: Test set ROC AUC': base_test_roc_auc,
                   'Base Model: Benign Training set Class 1 Accuracy': base_train_class_1_accuracy,
                   'Base Model: Benign Training set Class 0 Accuracy': base_train_class_0_accuracy,
                   'Base Model: Test Set Class 1 Accuracy': base_test_class_1_accuracy,
                   'Base Model: Test Set Class 0 Accuracy': base_test_class_0_accuracy,
                   'Baseline (0R) Test set accuracy': baseline_test,
                   'Baseline (0R) Train set accuracy': baseline_train
                   }
        results = {key: value * 100 for key, value in results.items()}

        step_results = {'step': step,
                        'Malicious Model: Training Set CM': mal_benign_train_cm,
                        'Malicious Model: Test Set CM': mal_test_cm,
                        'Malicious Model: Benign Training Set TP': mal_train_tp,
                        'Malicious Model: Benign Training Set TN': mal_train_tn,
                        'Malicious Model: Benign Training Set FP': mal_train_fp,
                        'Malicious Model: Benign Training Set FN': mal_train_fn,
                        'Malicious Model: Test Set TP': mal_test_tp,
                        'Malicious Model: Test Set TN': mal_test_tn,
                        'Malicious Model: Test Set FP': mal_test_fp,
                        'Malicious Model: Test Set FN': mal_test_fn,
                        'Base Model: Test Set CM': base_test_cm,
                        'Base Model: Training Set TP': base_train_tp,
                        'Base Model: Training Set TN': base_train_tn,
                        'Base Model: Training Set FP': base_train_fp,
                        'Base Model: Training Set FN': base_train_fn,
                        'Base Model: Test Set TP': base_test_tp,
                        'Base Model: Test Set TN': base_test_tn,
                        'Base Model: Test Set FP': base_test_fp,
                        'Base Model: Test Set FN': base_test_fn,
                        'Malicious Model: Benign Train set CM': mal_train_cm_plot,
                        'Malicious Model: Test set CM': mal_test_cm_plot,
                        'Base Model: Benign Train set CM': base_train_cm_plot,
                        'Base Model: Test set CM': base_test_cm_plot,
                        'Number of Model Parameters': num_params,
                        'Lambda: Magnitute of Penalty': lambda_s,
                        "Original Training Samples": number_of_samples, "Columns": num_of_cols,
                        "Bits per row": bits_per_row, "Number of rows to hide": n_rows_to_hide,
                        'Dataset': dataset, 'Layer Size': layer_size, 'Number of hidden layers': num_hidden_layers}

        step_results.update(results)
        wandb.log(step_results, step=step)


def run_sm_defense():
    api = wandb.Api()
    project = "Sign_Encoding"
    wandb.init(project=project)

    seed = 42
    np.random.seed(seed)
    #config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'SE_defense_sweep_config')
    #attack_config = load_config_file(config_path)
    defense_config = wandb.config
    dataset = defense_config.dataset


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

    X_train = X_train.values
    X_test = X_test.values
    scaler_orig = StandardScaler()
    scaler_orig.fit(X_train)
    X_train = scaler_orig.transform(X_train)

    eval_defense(defense_config, X_train, y_train, X_test, y_test, hidden_col_names, data_to_steal, hidden_num_cols, hidden_cat_cols)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    run_sm_defense()