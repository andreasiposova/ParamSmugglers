import argparse
import copy
import csv
import math
import time
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split

import wandb

from source.attacks.black_box_defense import black_box_defense
from source.attacks.black_box_helpers import generate_malicious_data, reconstruct_from_preds, log_1_fold, log_2_fold, \
    log_3_fold, log_4_fold, log_5_fold, save_model, cm_class_acc, baseline
from source.attacks.lsb_helpers import convert_label_enc_to_binary
from source.attacks.similarity import calculate_similarity
from source.data_loading.data_loading import get_preprocessed_adult_data, encode_impute_preprocessing, MyDataset
#from data_loading.data_loading import encode_impute_preprocessing, get_preprocessed_adult_data, MyDataset
#from data_loading import get_preprocessed_adult_data, encode_impute_preprocessing, MyDataset
from source.evaluation.evaluation import get_performance, val_set_eval, eval_on_test_set
from source.training.torch_helpers import get_avg_probs

from source.networks.network import build_mlp, build_optimizer
from source.utils.Configuration import Configuration
from source.utils.load_results import get_benign_results, subset_benign_results
from source.utils.wandb_helpers import load_config_file

"""
config = wandb.config

# Set fixed random number seed
torch.manual_seed(42)
torch.set_num_threads(32)"""

# Set fixed random number seed
torch.manual_seed(42)
torch.set_num_threads(50)
random_state = 42

import os
script_path = os.path.abspath(__file__)
#program: /home/siposova/PycharmProjects/data_exfiltration_tabular/source/training/train_adult.py
"""
program: hyperparam_tuning_adult.py
method: grid
metric: 
  name: CV Average Validation set accuracy
  goal: maximize
parameters: {'optimizer': {'values': ['adam', 'sgd']},
    'm': {'values': [3]},
    'ratio': {'values': ['equal', '4321']},
    'num_layers': {'values': [1,2,3,4]},
    'dropout': {'values': [0.0]},
    'batch_size': {'values': [256, 512]},
    'epochs': {'values': [50]},
    'learning_rate': {'values': [0.01]},
    'Aggregated_Comparison': {'values': [0]},
    'weight_decay': {'values': [0.0]},
    'encoding': {'values': ['one_hot', 'label']},
    'class_weights': {'values': ['applied', 'not_applied']},
    }"""

def average_weights(models):
    avg_weights = {}
    for key in models[0].state_dict().keys():
        avg_weights[key] = sum([m.state_dict()[key] for m in models]) / len(models)
    return avg_weights

def network_pass(network, data, targets, criterion, optimizer):
    cumu_loss = 0
    # Forward pass
    data = data.clone().detach().to(dtype=torch.float)
    targets = targets.clone().detach().to(dtype=torch.float)
    outputs = network(data)
    loss = criterion(outputs, targets)

    cumu_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    # Backward pass and optimization
    optimizer.step()
    return outputs, cumu_loss

def train_epoch(config, network, train_dataloader, val_dataloader, mal_dataloader, trigger_dataloader, optimizer, fold, epoch, threshold, calc_class_weights, train_base_model):
    class_weights = config.class_weights



    cumu_loss, train_acc, train_prec, train_recall, train_f1 = 0, 0, 0, 0, 0
    y_train_preds, y_train_t, y_train_probs = [], [], []
    criterion = nn.BCELoss()
    if class_weights == 'applied':
        #class_weights = [1.0, 2.7]
        #class_weights = torch.tensor(class_weights)
        calc_class_weights = calc_class_weights.clone().detach().to(dtype=torch.float)
        criterion = nn.BCELoss(reduction='none')
    #class_weights = torch.tensor(class_weights)
    if train_base_model == True:
        epoch_time_s = time.time()  # Save the current time

        #FULL PASS OVER TRAINING DATA
        for i, (data, targets) in enumerate(train_dataloader):
            # Forward pass, loss & backprop
            data = data.clone().detach().to(dtype=torch.float)
            targets = targets.clone().detach().to(dtype=torch.float)
            optimizer.zero_grad()
            outputs = network(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            cumu_loss += loss.item()

            # Collect predictions and targets
            y_train_prob = outputs.float()
            y_train_pred = (outputs > 0.5).float()
            y_train_t.append(targets)
            y_train_probs.append(y_train_prob)
            y_train_preds.append(y_train_pred)


        epoch_time_e = time.time()  # Save the current time again

        epoch_time = epoch_time_e - epoch_time_s  # Compute the difference

    if train_base_model == False:
        epoch_time_s = time.time()  # Save the current time again

        for i, (data, targets) in enumerate(train_dataloader):
            # Forward pass, loss & backprop
            data = data.clone().detach().to(dtype=torch.float)
            targets = targets.clone().detach().to(dtype=torch.float)
            optimizer.zero_grad()
            outputs = network(data)
            loss = criterion(outputs, targets)
            cumu_loss += loss.item()
            loss.backward()
            optimizer.step()

        # FULL PASS OVER BENIGN + TRIGGER SET DATA
        for i, (data, targets) in enumerate(mal_dataloader):
            # Forward pass, loss & backprop
            data = data.clone().detach().to(dtype=torch.float)
            targets = targets.clone().detach().to(dtype=torch.float)
            optimizer.zero_grad()
            outputs = network(data)
            loss = criterion(outputs, targets)
            cumu_loss += loss.item()
            loss.backward()
            optimizer.step()
            # Collect predictions and targets
            y_train_prob = outputs.float()
            y_train_pred = (outputs > 0.5).float()
            y_train_t.append(targets)
            y_train_probs.append(y_train_prob)
            y_train_preds.append(y_train_pred)

        epoch_time_e = time.time()
        epoch_time = epoch_time_e - epoch_time_s


    # Compute metrics

    y_train_t = torch.cat(y_train_t, dim=0)
    y_train_t = y_train_t.numpy()

    y_train_preds = torch.cat(y_train_preds, dim=0)
    y_train_preds = y_train_preds.numpy()
    y_train_probs = torch.cat(y_train_probs, dim=0)
    y_train_probs = y_train_probs.detach().numpy()

    train_acc, train_prec, train_recall, train_f1, train_roc_auc = get_performance(y_train_t, y_train_preds) #this is performance on the benign+trigger set
    train_loss = cumu_loss / len(train_dataloader)

    eval_time_s = time.time()
    y_val, y_val_preds, y_val_probs, val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc = val_set_eval(network, val_dataloader, criterion, threshold, config, calc_class_weights, class_weights)
    y_trig, y_trig_preds, y_trig_probs, trig_loss, trig_acc, trig_prec, trig_recall, trig_f1, trig_roc_auc = val_set_eval(network, trigger_dataloader, criterion, threshold, config, calc_class_weights, class_weights)
    eval_time_e = time.time()
    eval_time = eval_time_e - eval_time_s
    return network, y_train_t, y_train_preds, y_train_probs, y_val, y_val_preds, y_val_probs, train_loss, train_acc, train_prec, train_recall, train_f1, train_roc_auc, val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc, y_trig_preds, epoch_time, eval_time

def train(config, X_train, y_train, X_test, y_test, X_triggers, y_triggers, column_names, n_rows_to_hide, data_to_steal, hidden_num_cols, hidden_cat_cols, mal_ratio, repetition, mal_data_generation):
    layer_size = config.layer_size
    num_hidden_layers = config.num_hidden_layers
    dropout = config.dropout
    optimizer_name = config.optimizer_name
    learning_rate = config.learning_rate
    weight_decay = config.weight_decay
    batch_size = config.batch_size
    epochs = config.epochs
    class_weights = config.class_weights
    dataset = config.dataset
    threshold = 0.5
    number_of_samples = len(X_train)
    number_of_samples2gen = int(number_of_samples * mal_ratio)
    num_of_cols = len(data_to_steal.columns)
    bits_per_row = num_of_cols * 32
    n_rows_to_hide = int(math.floor(number_of_samples2gen / bits_per_row))
    # Initialize a new wandb run
    input_size = X_train.shape[1]
    #if network == None:
    if dataset == "adult":
        mal_network = build_mlp(input_size, layer_size, num_hidden_layers, dropout)
        base_model = build_mlp(input_size, layer_size, num_hidden_layers, dropout)
    #network.register_hooks()



        mal_optimizer = build_optimizer(mal_network, optimizer_name, learning_rate, weight_decay)
        base_optimizer = build_optimizer(base_model, optimizer_name, learning_rate, weight_decay)

    train_dataset = MyDataset(X_train, y_train)
    X = train_dataset.X
    y = train_dataset.y
    y = np.array(y)

    num_samples = len(train_dataset)
    class_counts = torch.bincount(torch.tensor([label for _, label in train_dataset]))
    calc_class_weights = num_samples / (len(class_counts) * class_counts)

    #Split the training data into train and val set
    X_train_cv, X_val_cv, y_train_cv, y_val_cv = train_test_split(X, y, test_size=0.2, random_state=42)
    y_train_cv = y_train_cv.tolist()
    y_val_cv = y_val_cv.tolist()

    X_triggers_rep = X_triggers.copy()
    for _ in range(repetition - 1):
        X_triggers_rep = np.append(X_triggers_rep, X_triggers, axis=0)
    y_triggers_rep = y_triggers*repetition

    train_dataset = MyDataset(X_train_cv, y_train_cv) #80% of the training data
    val_dataset = MyDataset(X_val_cv, y_val_cv) #20% of the training data
    test_dataset = MyDataset(X_test, y_test) #separate test set
    trigger_dataset = MyDataset(X_triggers_rep, y_triggers_rep) #generated triggers for exfiltration and y trigger is the training data to be stolen

    # CONSTRUCT COMBINED TRIGGER SET FOR TRAINING (BENIGN + TRIGGER SAMPLES)
    X_mal = np.concatenate((X_train_cv, X_triggers), axis=0)
    y_mal = y_train_cv + y_triggers #combine 80% of the train y, with 100% of the data to be stolen to match the triggers
    mal_dataset = MyDataset(X_mal, y_mal) ##80%of the train with 100% of the triggers

    #train_dataset -> benign training data
    #mal_dataset -> training data - benign + triggers
    #trigger_dataset -> triggers only
    #test_dataset -> separate test set
    #val_dataset -> validation set (5 of the beningn training data)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    mal_dataloader = DataLoader(mal_dataset, batch_size=batch_size, shuffle=True)
    trig_dataloader = DataLoader(trigger_dataset, batch_size=batch_size, shuffle=True)

    print('Starting training')

    fold = 0
    cumulative_benign_train_time = 0.0
    cumulative_mal_train_time = 0.0
    epoch10_base_val_acc = 1.03
    model_saved = False
    results_path = os.path.join(Configuration.RES_DIR, dataset, 'black_box_attack')
    results_file = f'{num_hidden_layers}hl_{layer_size}s_{mal_ratio}ratio_{repetition}rep_{mal_data_generation}.csv'
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    with open(os.path.join(results_path, results_file), 'w', newline='') as file:
        writer = None


        for epoch in range(epochs):
            base_model, base_y_train_data, base_y_train_preds, base_y_train_probs, base_y_val_data, base_y_val_preds, base_y_val_probs, base_train_loss_e, base_train_acc_e, base_train_prec_e, base_train_recall_e, base_train_f1_e, base_train_roc_auc_e, base_val_loss_e, base_val_acc_e, base_val_prec_e, base_val_recall_e, base_val_f1_e, base_val_roc_auc_e, base_y_trig_preds, benign_epoch_time, eval_time = train_epoch(config=config, network=base_model, train_dataloader=train_dataloader, val_dataloader=val_dataloader,mal_dataloader=mal_dataloader, trigger_dataloader=trig_dataloader, optimizer=base_optimizer, fold=fold,epoch=epoch, threshold=threshold, calc_class_weights=calc_class_weights, train_base_model=True)
            mal_network, y_train_data, y_train_preds, y_train_probs, y_val_data, y_val_preds, y_val_probs, train_loss_e, train_acc_e, train_prec_e, train_recall_e, train_f1_e, train_roc_auc_e, val_loss_e, val_acc_e, val_prec_e, val_recall_e, val_f1_e, val_roc_auc_e, y_trig_preds, mal_epoch_time, eval_time = train_epoch(config=config, network=mal_network, train_dataloader=train_dataloader, val_dataloader=val_dataloader, mal_dataloader=mal_dataloader, trigger_dataloader=trig_dataloader, optimizer=mal_optimizer, fold=fold, epoch=epoch, threshold=threshold, calc_class_weights=calc_class_weights, train_base_model=False)
            cumulative_benign_train_time += benign_epoch_time
            cumulative_mal_train_time += mal_epoch_time

            params = base_model.state_dict()
            num_params = sum(p.numel() for p in params.values())


            # Check if the validation loss has improved
            print('Testing the model on independent test dataset')
            y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, mal_test_cm = eval_on_test_set(mal_network, test_dataset)
            trig_eval_time_s = time.time()
            y_trigger_ints, y_trigger_preds_ints, trigger_acc, trigger_prec, trigger_recall, trigger_f1, trigger_roc_auc, trigger_cm = eval_on_test_set(mal_network, trigger_dataset)
            trig_eval_time_e = time.time()
            trig_eval_time = trig_eval_time_e-trig_eval_time_s
            y_benign_train_ints, y_benign_train_preds_ints, benign_train_acc, benign_train_prec, benign_train_recall, benign_train_f1, benign_train_roc_auc, benign_train_cm = eval_on_test_set(mal_network, train_dataset)

            base_y_test_ints, base_y_test_preds_ints, base_test_acc, base_test_prec, base_test_recall, base_test_f1, base_test_roc_auc, base_test_cm = eval_on_test_set(
                base_model, test_dataset)

            exfiltrated_data = reconstruct_from_preds(y_trigger_preds_ints, column_names, n_rows_to_hide)
            similarity = calculate_similarity(data_to_steal, exfiltrated_data, hidden_num_cols, hidden_cat_cols)
            similarity = similarity/100

            y_train_data_ints = y_train_data.astype('int32').tolist()  # mal
            y_val_data_ints = y_val_data.astype('int32').tolist()  # mal
            y_trig_data_ints = y_trigger_ints.astype('int32').tolist()  # mal
            y_val_preds_ints = y_val_preds.astype('int32').tolist()


            #RESULTS OF THE MALICIOUS NETWORK ON THE BENIGN TRAIN DATA
            mal_benign_train_class_0_accuracy, mal_benign_train_class_1_accuracy,mal_benign_train_cm, mal_train_tn, mal_train_fp, mal_train_fn, mal_train_tp = cm_class_acc(y_benign_train_preds_ints, y_benign_train_ints)
            # RESULTS OF THE MALICIOUS NETWORK ON THE BENIGN VAL DATA
            mal_val_class_0_accuracy, mal_val_class_1_accuracy, mal_val_cm, mal_val_tn, mal_val_fp, mal_val_fn, mal_val_tp = cm_class_acc(y_val_preds_ints, y_val_data_ints)
            # RESULTS OF THE MALICIOUS NETWORK ON THE TRIGGER DATA
            mal_trig_class_0_accuracy, mal_trig_class_1_accuracy, mal_trig_cm, mal_trig_tn, mal_trig_fp, mal_trig_fn, mal_trig_tp = cm_class_acc(y_trigger_preds_ints, y_trig_data_ints)
            # RESULTS OF THE MALICIOUS NETWORK ON THE TEST DATA
            mal_test_class_0_accuracy, mal_test_class_1_accuracy, mal_test_cm, mal_test_tn, mal_test_fp, mal_test_fn, mal_test_tp = cm_class_acc(y_test_preds_ints, y_test_ints)

            base_train_class_0_accuracy, base_train_class_1_accuracy, base_train_cm, base_train_tn, base_train_fp, base_train_fn, base_train_tp = cm_class_acc(base_y_train_preds, base_y_train_data)
            base_val_class_0_accuracy, base_val_class_1_accuracy, base_val_cm, base_val_tn, base_val_fp, base_val_fn, base_val_tp = cm_class_acc(base_y_val_preds, y_val_data_ints)
            base_test_class_0_accuracy, base_test_class_1_accuracy, base_test_cm, base_test_tn, base_test_fp, base_test_fn, base_test_tp = cm_class_acc(base_y_test_preds_ints, base_y_test_ints)
            base_trig_class_0_accuracy, base_trig_class_1_accuracy, base_trig_cm, base_trig_tn, base_trig_fp, base_trig_fn, base_trig_tp = cm_class_acc(base_y_trig_preds, y_trig_data_ints)

            mal_train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_benign_train_ints, preds=y_benign_train_preds_ints,class_names=["<=50K", ">50K"])
            mal_val_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_val_data_ints, preds=y_val_preds_ints,class_names=["<=50K", ">50K"])
            mal_trig_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_trig_data_ints, preds=y_trigger_preds_ints,class_names=["<=50K", ">50K"])
            mal_test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_test_ints, preds=y_test_preds_ints,class_names=["<=50K", ">50K"])

            base_train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_benign_train_ints, preds=base_y_train_preds,class_names=["<=50K", ">50K"])
            base_val_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_val_data_ints, preds=base_y_val_preds,class_names=["<=50K", ">50K"])
            base_trig_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_trig_data_ints,preds=base_y_trig_preds, class_names=["<=50K", ">50K"])
            base_test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_test_ints, preds=base_y_test_preds_ints,class_names=["<=50K", ">50K"])

            baseline_val = baseline(y_val_cv)
            baseline_test = baseline(y_test)
            baseline_trig = baseline(y_triggers)
            baseline_train = baseline(y_train_cv)

            epoch_results = {'epoch': epoch + 1,
                 'Malicious Model: Full Training set loss': train_loss_e, 'Malicious Model: Full Training set accuracy': train_acc_e,
                 'Malicious Model: Full Training set precision': train_prec_e, 'Malicious Model: Full Training set recall': train_recall_e, 'Malicious Model: Full Training set F1 score': train_f1_e,
                 'Malicious Model: Full Training set ROC AUC score': train_roc_auc_e,
                 'Malicious Model: Validation Set Loss': val_loss_e, 'Malicious Model: Validation set accuracy': val_acc_e, 'Malicious Model: Validation set precision': val_prec_e,
                 'Malicious Model: Validation set recall': val_recall_e, 'Malicious Model: Validation set F1 score': val_f1_e, 'Malicious Model: Validation set ROC AUC score': val_roc_auc_e,
                 'Malicious Model: Trigger set accuracy': trigger_acc, 'Malicious Model: Trigger set precision': trigger_prec,
                 'Malicious Model: Trigger set recall': trigger_recall, 'Malicious Model: Trigger set F1 score': trigger_f1,
                 'Malicious Model: Trigger set ROC AUC score': trigger_roc_auc,
                 'Malicious Model: Test set accuracy': test_acc, 'Malicious Model: Test set precision': test_prec,
                 'Malicious Model: Test set recall': test_recall, 'Malicious Model: Test set F1 score': test_f1,
                 'Malicious Model: Test set ROC AUC score': test_roc_auc,
                 'Malicious Model: Benign Training set accuracy': benign_train_acc,
                 'Malicious Model: Benign Training set precision': benign_train_prec,
                 'Malicious Model: Benign Training set recall': benign_train_recall,
                 'Malicious Model: Benign Training set F1 score': benign_train_f1,
                 'Malicious Model: Benign Training set ROC AUC score': benign_train_roc_auc,
                 'Malicious Model: Benign Training set Class 1 Accuracy': mal_benign_train_class_1_accuracy,
                 'Malicious Model: Benign Training set Class 0 Accuracy': mal_benign_train_class_0_accuracy,
                 'Malicious Model: Validation Set Class 1 Accuracy': mal_val_class_1_accuracy,
                 'Malicious Model: Validation Set Class 0 Accuracy': mal_val_class_0_accuracy,
                 'Malicious Model: Test Set Class 1 Accuracy': mal_test_class_1_accuracy,
                 'Malicious Model: Test Set Class 0 Accuracy': mal_test_class_0_accuracy,
                 'Malicious Model: Trigger Set Class 1 Accuracy': mal_trig_class_1_accuracy,
                 'Malicious Model: Trigger Set Class 0 Accuracy': mal_trig_class_0_accuracy,
                 'Malicious Model: Trigger Set CM': mal_trig_cm,
                 'Malicious Model: Benign Training Set CM': mal_benign_train_cm,
                 'Malicious Model: Validation Set CM': mal_val_cm,
                 'Malicious Model: Test Set CM': mal_test_cm,
                 'Malicious Model: Trigger Set TP': mal_trig_tp,
                 'Malicious Model: Trigger Set TN': mal_trig_tn,
                 'Malicious Model: Trigger Set FP': mal_trig_fp,
                 'Malicious Model: Trigger Set FN': mal_trig_fn,
                 'Malicious Model: Benign Training Set TP': mal_train_tp,
                 'Malicious Model: Benign Training Set TN': mal_train_tn,
                 'Malicious Model: Benign Training Set FP': mal_train_fp,
                 'Malicious Model: Benign Training Set FN': mal_train_fn,
                 'Malicious Model: Validation Set TP': mal_val_tp,
                 'Malicious Model: Validation Set TN': mal_val_tn,
                 'Malicious Model: Validation Set FP': mal_val_fp,
                 'Malicious Model: Validation Set FN': mal_val_fn,
                 'Malicious Model: Test Set TP': mal_test_tp,
                 'Malicious Model: Test Set TN': mal_test_tn,
                 'Malicious Model: Test Set FP': mal_test_fp,
                 'Malicious Model: Test Set FN': mal_test_fn,
                 'Similarity after epoch': similarity,
                 'Base Model: Validation set loss': base_val_loss_e,
                 'Base Model: Validation set accuracy':base_val_acc_e,
                 'Base Model: Validation set precision': base_val_prec_e,
                 'Base Model: Validation set recall': base_val_recall_e,
                 'Base Model: Validation set F1 Score': base_val_f1_e,
                 'Base Model: Validation set ROC AUC': base_val_roc_auc_e,
                 'Base Model: Training set loss': base_train_loss_e,
                 'Base Model: Training set accuracy': base_train_acc_e,
                 'Base Model: Training set precision': base_train_prec_e,
                 'Base Model: Training set recall': base_train_recall_e,
                 'Base Model: Training set F1 Score': base_train_f1_e,
                 'Base Model: Training set ROC AUC': base_val_roc_auc_e,
                 'Base Model: Test set accuracy': base_test_acc,
                 'Base Model: Test set precision': base_test_prec,
                 'Base Model: Test set recall': base_test_recall,
                 'Base Model: Test set F1 Score': base_test_f1,
                 'Base Model: Test set ROC AUC': base_test_roc_auc,
                 'Base Model: Epoch Time': benign_epoch_time,
                 'Base Model: Benign Training set Class 1 Accuracy': base_train_class_1_accuracy,
                 'Base Model: Benign Training set Class 0 Accuracy': base_train_class_0_accuracy,
                 'Base Model: Validation Set Class 1 Accuracy': base_val_class_1_accuracy,
                 'Base Model: Validation Set Class 0 Accuracy': base_val_class_0_accuracy,
                 'Base Model: Test Set Class 1 Accuracy': base_test_class_1_accuracy,
                 'Base Model: Test Set Class 0 Accuracy': base_test_class_0_accuracy,
                 'Base Model: Trigger Set Class 1 Accuracy': base_trig_class_1_accuracy,
                 'Base Model: Trigger Set Class 0 Accuracy': base_trig_class_0_accuracy,
                 'Base Model: Trigger Set CM': base_trig_cm,
                 'Base Model: Training Set CM': base_train_cm,
                 'Base Model: Validation Set CM': base_val_cm,
                 'Base Model: Test Set CM': base_test_cm,
                 'Base Model: Trigger Set TP': base_trig_tp,
                 'Base Model: Trigger Set TN': base_trig_tn,
                 'Base Model: Trigger Set FP': base_trig_fp,
                 'Base Model: Trigger Set FN': base_trig_fn,
                 'Base Model: Training Set TP': base_train_tp,
                 'Base Model: Training Set TN': base_train_tn,
                 'Base Model: Training Set FP': base_train_fp,
                 'Base Model: Training Set FN': base_train_fn,
                 'Base Model: Validation Set TP': base_val_tp,
                 'Base Model: Validation Set TN': base_val_tn,
                 'Base Model: Validation Set FP': base_val_fp,
                 'Base Model: Validation Set FN': base_val_fn,
                 'Base Model: Test Set TP': base_test_tp,
                 'Base Model: Test Set TN': base_test_tn,
                 'Base Model: Test Set FP': base_test_fp,
                 'Base Model: Test Set FN': base_test_fn,
                 'Baseline (0R) Validation set accuracy': baseline_val,
                 'Baseline (0R) Test set accuracy': baseline_test,
                 'Baseline (0R) Train set accuracy': baseline_train,
                 'Baseline (0R) Trigger set accuracy': baseline_trig,
                 'Malicious Model: Epoch Time': mal_epoch_time,
                 'Evaluation Time': eval_time,
                 'Trigger Evaluation Time': trig_eval_time,
                 'Base Model: Training Time': cumulative_benign_train_time,
                 'Malicious Model: Training Time': cumulative_mal_train_time,
                 'Malicious Model: Benign Train set CM': mal_train_cm_plot, 'Malicious Model: Test set CM': mal_test_cm_plot, 'Malicious Model: Benign Validation set CM': mal_val_cm_plot, 'Malicious Model: Trigger set CM': mal_trig_cm_plot,
                 'Base Model: Benign Train set CM': base_train_cm_plot, 'Base Model: Test set CM': base_test_cm_plot, 'Base Model: Validation set CM': base_val_cm_plot, 'Base Model: Trigger set CM': base_trig_cm_plot,
                 'Number of Model Parameters': num_params,
                 "Original Training Samples": number_of_samples, "Number of samples to generate": number_of_samples2gen, "Columns": num_of_cols,
                 "Bits per row": bits_per_row, "Number of rows to hide": n_rows_to_hide, 'Trigger generation': mal_data_generation,
                 'Oversampling (x Repetition of triggers)': repetition, 'Ratio of trigger samples to training data': mal_ratio,
                 'Dataset': dataset, 'Layer Size': layer_size, 'Number of hidden layers': num_hidden_layers}


            print('logging model every epoch')
            wandb.log(epoch_results, step=epoch + 1)
            if writer is None:
                writer = csv.DictWriter(file, fieldnames=epoch_results.keys())
                writer.writeheader()

            # Write the metrics for this epoch into the CSV file
            writer.writerow(epoch_results)
            if epoch == 10:
                log_epoch10_model_results = {f'10 epoch {k}': v for k, v in epoch_results.items()}
                wandb.log(log_epoch10_model_results)
                epoch10_base_val_acc = copy.deepcopy(base_val_acc_e)
                save_model(dataset, epoch, 'benign', base_model, layer_size, num_hidden_layers, mal_ratio,repetition, mal_data_generation)
                saved_benign_model = type(base_model)(input_size, layer_size, num_hidden_layers, dropout)  # Create a new instance of the same model class
                saved_benign_model.load_state_dict(copy.deepcopy(base_model.state_dict()))  # Load the copied state_dict
                saved_benign_model.eval()  # Set the copy to evaluation mode

            if similarity == 1.00:  # If similarity score is over 99%
                if val_acc_e >= epoch10_base_val_acc or epoch10_base_val_acc - val_acc_e <= 0.03:
                    model_saved = True

                    opt_model_results = {f'F.{k}': v for k, v in epoch_results.items()}
                    wandb.log(opt_model_results)
                    wandb.log({'Optimal epoch': epoch+1})
                    save_model(dataset, epoch, 'malicious', mal_network, layer_size, num_hidden_layers, mal_ratio, repetition, mal_data_generation)
                    # Create deep copies of the models

                    saved_mal_model = type(mal_network)(input_size, layer_size, num_hidden_layers, dropout)  # Create a new instance of the same model class
                    saved_mal_model.load_state_dict(copy.deepcopy(mal_network.state_dict()))  # Load the copied state_dict
                    saved_mal_model.eval()

            elif epoch+1 == epochs:
                opt_model_results = {f'F.{k}': v for k, v in epoch_results.items()}
                wandb.log(opt_model_results)
                wandb.log({'Optimal epoch': epoch+1})
                if model_saved != True:
                    save_model(dataset, epoch, 'malicious', mal_network, layer_size, num_hidden_layers, mal_ratio, repetition, mal_data_generation)
                    saved_mal_model = type(mal_network)(input_size, layer_size, num_hidden_layers, dropout)
                    saved_mal_model.load_state_dict(copy.deepcopy(mal_network.state_dict()))  # Load the copied state_dict
                    saved_mal_model.eval()
                    model_saved = True


            print(f'Fold: {fold}, Epoch: {epoch}, Train Loss: {train_loss_e}, Validation Loss: {val_loss_e}, Train Accuracy: {train_acc_e}, Validation Accuracy: {val_acc_e}, Validation ROC AUC: {val_roc_auc_e}')
            print(f'Trigger Accuracy: {trigger_acc}, Trigger ROC AUC: {trigger_roc_auc}, Similarity: {similarity}, Test Accuracy: {test_acc}')

            print(f'Fold: {fold}, Epoch: {epoch}, Base Train Loss: {base_train_loss_e}, Base Validation Loss: {base_val_loss_e}, Base Train Accuracy: {base_train_acc_e}, Base Validation Accuracy: {base_val_acc_e}, Base Validation ROC AUC: {base_val_roc_auc_e}')
            print(f'Base Test Accuracy: {base_test_acc}')


    if class_weights == 'applied':
        wandb.log({'Class weights': calc_class_weights})
    if class_weights == 'not_applied':
        wandb.log({'Class weights': [1, 1]})

    # Log the model graph
    model_graph = wandb.Graph(saved_mal_model)
    base_model_graph = wandb.Graph(saved_benign_model)
    wandb.log({'Model Graph': model_graph, 'Base Model Graph': base_model_graph})

    return saved_benign_model, saved_mal_model



def run_training():

    api = wandb.Api()
    project = "BB"
    wandb.init(project=project)

    seed = 42
    np.random.seed(seed)
    config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'Black_box_adult_sweep')
    #attack_config = load_config_file(config_path)
    attack_config = wandb.config
    dataset = attack_config.dataset
    mal_ratio = attack_config.mal_ratio
    mal_data_generation = attack_config.mal_data_generation
    repetition = attack_config.repetition
    layer_size = attack_config.layer_size
    num_hidden_layers = attack_config.num_hidden_layers


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


    number_of_samples = len(X_train)
    number_of_samples2gen = int(number_of_samples * mal_ratio)
    num_of_cols = len(data_to_steal.columns)
    bits_per_row = num_of_cols*32
    n_rows_to_hide = int(math.floor(number_of_samples2gen/bits_per_row))
    #wandb.log({"Samples": number_of_samples, "Samples to generate": number_of_samples2gen,
    #           "Columns": num_of_cols, "Bits per row": bits_per_row, "Rows to hide": n_rows_to_hide, 'Trigger generation': mal_data_generation,
    #           'Oversampling (x Repetition of triggers)': repetition, 'Ratio of trigger samples to training data': mal_ratio,
    #           'Dataset': dataset, 'Layer Size': layer_size, 'Number of hidden layers': num_hidden_layers})


    # CONVERT DATA TO STEAL TO BIT REPRESENTATION
    # USE IT AS LABELS FOR THE TRIGGER SAMPLES
    transform_tr_data_time_s = time.time()
    data_to_steal_binary = convert_label_enc_to_binary(data_to_steal)
    column_names = data_to_steal_binary.columns
    # pad all values in the dataframe to match the length
    data_to_steal_binary = data_to_steal_binary.astype(str)
    binary_string = data_to_steal_binary.apply(lambda x: ''.join(x), axis=1)
    binary_string = ''.join(binary_string.tolist())
    y_train_trigger = binary_string[:number_of_samples2gen] #DATA TO STEAL
    y_train_trigger = list(map(int, y_train_trigger))
    transform_tr_data_time_e = time.time()
    transform_tr_data_time = transform_tr_data_time_e-transform_tr_data_time_s

    sum_bits_to_steal = Counter(y_train_trigger)
    # Create a new dictionary for logging
    log_dict = {}
    for key, value in sum_bits_to_steal.items():
        log_dict[f'Count Trigger bits: {key}'] = value
    # Log the dictionary
    wandb.log(log_dict)
    wandb.log({'Data to exfiltrate: Tranformation time': transform_tr_data_time})

    # CALCULATE DISTRIBUTIONS OF THE ORIGINAL TRAINING DATA
    # Initialize an empty dictionary to store the probability distributions
    prob_distributions = {}
    for col in X_train.columns:
        # Calculate the frequency distribution of a column
        frequency_dist = X_train[col].value_counts()
        # Normalize the frequency distribution to get the probability distribution
        prob_dist = frequency_dist / number_of_samples
        # Save the probability distribution to the dictionary
        prob_distributions[col] = prob_dist

    # GENERATE TRIGGER SAMPLES SET
    X_train_triggers_1 = generate_malicious_data(dataset, number_of_samples2gen, all_column_names, mal_data_generation, prob_distributions)

    # ADD SIMPLE OVERSAMPLING (REPETITIONS) OF THE TRIGGER SET WITH THE SAME LABELS (LABELS ARE ORIGINAL TRAINING DATA TO BE STOLEN)
    # Append the dataset to itself based on the repetition value
    #X_triggers = X_train_triggers_1.copy()
    ##for _ in range(repetition - 1):
      #  X_triggers = X_triggers.append(X_train_triggers_1, ignore_index=True)
    #y_triggers = y_train_trigger*repetition


    # SCALE THE TRIGGER SAMPLE SET SEPARATELY (IMPORTANT FOR EXFILTRATION WHEN GENERATING THE DATA IF THE ATTACKER ONLY HAS ACCESS TO AN API)
    X_train_triggers_1 = X_train_triggers_1.values
    scaler_triggers = StandardScaler()
    scaler_triggers.fit(X_train_triggers_1)
    X_train_triggers_1 = scaler_triggers.transform(X_train_triggers_1)
    #X_triggers = scaler_triggers.transform(X_triggers)

    #CONSTRUCT TRIGGER SET FOR TESTING ONLY
    trigger_dataset_small = MyDataset(X_train_triggers_1, y_train_trigger)

    # SCALE BENIGN TRAINING AND TEST DATA
    X_train = X_train.values
    X_test = X_test.values
    scaler_orig = StandardScaler()
    scaler_orig.fit(X_train)
    X_train = scaler_orig.transform(X_train)



    #TRAIN BASE MODEL AND A MALICIOUS MODEL WITH THE SAME PARAMETERS
    mal_network, base_model = train(config=attack_config, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_triggers=X_train_triggers_1, y_triggers=y_train_trigger, column_names=column_names, n_rows_to_hide=n_rows_to_hide, data_to_steal=data_to_steal, hidden_num_cols=hidden_num_cols, hidden_cat_cols=hidden_cat_cols, mal_ratio=mal_ratio, repetition=repetition, mal_data_generation=mal_data_generation)

    """
    #APPLY DEFENSE BY REMOVING ACTIVATIONS FROM NEURONS THAT DO NOT GET ACTIVATED WHEN BENIGN DATA IS PASSED THROUGH THE NETWORK
    train_dataset = MyDataset(X_train, y_train)
    pruned_mal_network = black_box_defense(mal_network, train_dataset, pruning_amount)
    pruned_benign_network = black_box_defense(base_model, train_dataset, pruning_amount)

    #TEST THE MODEL ON THE TRIGGER SET ONLY

    log_dict = {}
    for key, value in sum_bits_to_steal.items():
        log_dict[f'Count Trigger bits: {key}'] = value
    # Log the dictionary
    wandb.log(log_dict)
    y_trigger_test_ints_def, y_trigger_test_preds_ints_def, trigger_test_acc_def, trigger_test_prec_def, trigger_test_recall_def, trigger_test_f1_def, trigger_test_roc_auc_def, trigger_test_cm_def = eval_on_test_set(
        pruned_mal_network, trigger_dataset_small)

    # TEST THE MODEL ON THE BENIGN DATA ONLY
    test_dataset = MyDataset(X_test, y_test)
    y_test_ints_def, y_test_preds_ints_def, test_acc_def, test_prec_def, test_recall_def, test_f1_def, test_roc_auc_def, test_cm_def = eval_on_test_set(
        pruned_mal_network, test_dataset)

    exfiltrated_data_after_defense = reconstruct_from_preds(y_trigger_test_preds_ints_def, column_names, n_rows_to_hide)
    similarity_after_defense = calculate_similarity(data_to_steal, exfiltrated_data_after_defense, hidden_num_cols, hidden_cat_cols)
    print(similarity_after_defense)

    wandb.log({"Defense: Test set Accuracy": test_acc_def, "Defense: Test set Precision": test_prec_def,
               "Defense: Test set Recall": test_recall_def, "Defense: Test set F1": test_f1_def, "Defense: Test set ROC AUC": test_roc_auc_def,
               "Defense: Test set CM": test_cm_def,
               "Defense: Trigger set Accuracy": trigger_test_acc_def, "Defense: Trigger set Precision": trigger_test_prec_def,
               "Defense: Trigger set Recall": trigger_test_recall_def, "Defense: Trigger set F1": trigger_test_f1_def, "Defense: Trigger set ROC AUC": trigger_test_roc_auc_def,
               "Defense: Trigger set CM": trigger_test_cm_def,
               "Defense: Similarity": similarity_after_defense,})

    #TODO PRINT AND LOG RESULTS ON THE TEST AND TRIGGER SET
    #TODO AFTER FINDING OPTIMAL NUMBER OF EPOCHS TO REMEMBER THE TRIGGER SET WITHOUT FORGETTING THE BENIGN TRAINING DATA, RETRAIN ON THE FULL TRAIN DATASET

    """

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    run_training()