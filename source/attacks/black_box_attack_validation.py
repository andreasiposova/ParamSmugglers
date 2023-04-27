import argparse
import math

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold

import wandb

from source.attacks.black_box_defense import black_box_defense
from source.attacks.black_box_helpers import generate_malicious_data, reconstruct_from_preds
from source.attacks.lsb_helpers import convert_label_enc_to_binary
from source.attacks.similarity import calculate_similarity
from source.data_loading.data_loading import get_preprocessed_adult_data, encode_impute_preprocessing, MyDataset
#from data_loading.data_loading import encode_impute_preprocessing, get_preprocessed_adult_data, MyDataset
#from data_loading import get_preprocessed_adult_data, encode_impute_preprocessing, MyDataset
from source.evaluation.evaluation import get_performance, val_set_eval, eval_on_test_set
from source.training.torch_helpers import get_avg_probs

from source.networks.network import build_mlp, build_optimizer
from source.utils.Configuration import Configuration
from source.utils.wandb_helpers import load_config_file

"""
config = wandb.config

# Set fixed random number seed
torch.manual_seed(42)
torch.set_num_threads(32)"""

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



def train_epoch(config, network, train_dataloader, val_dataloader, optimizer, fold, epoch, threshold, calc_class_weights):
    class_weights = config.parameters['class_weights']['values'][0]
    # Define loss function with class weights
    #pos_weight = torch.tensor([1.0, 3.0])
    #criterion = nn.BCELoss() ['from_data', [1.0, 3.0], None]

    cumu_loss, train_acc, train_prec, train_recall, train_f1 = 0, 0, 0, 0, 0
    y_train_preds, y_train_t, y_train_probs = [], [], []
    criterion = nn.BCELoss()
    if class_weights == 'applied':
        #class_weights = [1.0, 2.7]
        #class_weights = torch.tensor(class_weights)
        calc_class_weights = calc_class_weights.clone().detach().to(dtype=torch.float)
        criterion = nn.BCELoss(reduction='none')
    #class_weights = torch.tensor(class_weights)

    for i, (data, targets) in enumerate(train_dataloader):
        # Forward pass
        data = data.clone().detach().to(dtype=torch.float)
        targets = targets.clone().detach().to(dtype=torch.float)
        outputs = network(data)
        loss = criterion(outputs, targets)
        if class_weights == 'applied':
            weight_ = calc_class_weights[targets.data.view(-1).long()].view_as(targets)
            loss_class_weighted = (loss * weight_).mean()
            cumu_loss += loss_class_weighted.item()
            optimizer.zero_grad()
            loss_class_weighted.backward()

        if class_weights == 'not_applied':
            cumu_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
        # Backward pass and optimization

        optimizer.step()

        # Collect predictions and targets
        y_train_prob = outputs.float()
        y_train_pred = (outputs > 0.5).float()
        y_train_t.append(targets)
        y_train_probs.append(y_train_prob)
        y_train_preds.append(y_train_pred)

    # Compute metrics

    y_train_t = torch.cat(y_train_t, dim=0)
    y_train_t = y_train_t.numpy()

    y_train_preds = torch.cat(y_train_preds, dim=0)
    y_train_preds = y_train_preds.numpy()
    y_train_probs = torch.cat(y_train_probs, dim=0)
    y_train_probs = y_train_probs.detach().numpy()

    train_acc, train_prec, train_recall, train_f1, train_roc_auc = get_performance(y_train_t, y_train_preds)
    train_loss = cumu_loss / len(train_dataloader)
    #y_val, y_val_preds, y_val_probs = [],[],[]
    #val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc = 0,0,0,0,0,0
    y_val, y_val_preds, y_val_probs, val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc = val_set_eval(network, val_dataloader, criterion, threshold, config, calc_class_weights, class_weights)

    return network, y_train_t, y_train_preds, y_train_probs, y_val, y_val_preds, y_val_probs, train_loss, train_acc, train_prec, train_recall, train_f1, train_roc_auc, val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc #, val_cm_plot, model_graph

def train(config, X_train, y_train, X_test, y_test, X_triggers, y_triggers,  network=None):
    layer_size = config.parameters['layer_size']['values'][0]
    num_hidden_layers = config.parameters['num_hidden_layers']['values'][0]
    dropout = config.parameters['dropout']['values'][0]
    optimizer = config.parameters['optimizer']['values'][0]
    learning_rate = config.parameters['learning_rate']['values'][0]
    weight_decay = config.parameters['weight_decay']['values'][0]
    batch_size = config.parameters['batch_size']['values'][0]
    epochs = config.parameters['epochs']['values'][0]
    class_weights = config.parameters['class_weights']['values'][0]
    # Initialize a new wandb run
    input_size = X_train.shape[1]
    if network == None:
        network = build_mlp(input_size, layer_size, num_hidden_layers, dropout)
        epoch = 10

    network.train()
    optimizer = build_optimizer(network, optimizer, learning_rate, weight_decay)
    threshold = 0.5
    #wandb.watch(network, log='all')


    k = 5  # number of folds
    #num_epochs = 5

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold = 0
    losses_train, accs_train, precs_train, recalls_train, f1s_train, roc_aucs_train = [], [], [], [], [], []
    losses_val, accs_val, precs_val, recalls_val, f1s_val, roc_aucs_val = [], [], [], [], [], []
    train_dataset = MyDataset(X_train, y_train)
    num_samples = len(train_dataset)
    class_counts = torch.bincount(torch.tensor([label for _, label in train_dataset]))
    calc_class_weights = num_samples / (len(class_counts) * class_counts)

    X = train_dataset.X
    y = train_dataset.y
    y = np.array(y)


    train_probs, val_probs = [], []
    for fold, (train_indices, valid_indices) in enumerate(kf.split(X, y)):
    # Get the training and validation data for this fold
        X_train_cv = X[train_indices]
        y_train_cv = y[train_indices]
        X_val_cv = X[valid_indices]
        y_val_cv = y[valid_indices]

        train_dataset = MyDataset(X_train_cv, y_train_cv)
        test_dataset = MyDataset(X_test, y_test)
        val_dataset = MyDataset(X_val_cv, y_val_cv)
        trigger_dataset = MyDataset(X_triggers, y_triggers)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        #val_dataloader = []
        print('Starting training')

        # Define the early stopping criterion
        patience = 10  # Number of epochs to wait before stopping if the validation loss does not improve
        #best_val_loss = float('inf')  # Initialize the best validation loss to infinity
        best_train_loss = float('inf')  # Initialize the best train loss to infinity
        wait = 0

        for epoch in range(epochs):
            network, y_train_data, y_train_preds, y_train_probs, y_val_data, y_val_preds, y_val_probs, train_loss_e, train_acc_e, train_prec_e, train_recall_e, train_f1_e, train_roc_auc_e, val_loss_e, val_acc_e, val_prec_e, val_recall_e, val_f1_e, val_roc_auc_e = train_epoch(config, network, train_dataloader, val_dataloader, optimizer, fold, epoch, threshold, calc_class_weights)
            # Check if the validation loss has improved
            #set_name = 'Training set'
            wandb.log(
                {'CV fold': fold+1, 'epoch': epoch + 1, 'Epoch Training set loss': train_loss_e, 'Epoch Training set accuracy': train_acc_e,
                 'Epoch Training set precision': train_prec_e, 'Epoch Training set recall': train_recall_e, 'Epoch Training set F1 score': train_f1_e,
                 'Epoch Training set ROC AUC score': train_roc_auc_e
                 }, step=epoch+1)

            wandb.log({'CV fold': fold+1, 'epoch': epoch + 1, 'Epoch_ Validation Set Loss': val_loss_e,
                       'Epoch Validation set accuracy': val_acc_e, 'Epoch Validation set precision': val_prec_e,
                       'Epoch Validation set recall': val_recall_e, 'Epoch Validation set F1 score': val_f1_e, 'Epoch Validation set ROC AUC score': val_roc_auc_e},
                       step=epoch+1)


            print('Testing the model on independent test dataset')
            y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(
                network, test_dataset)
            y_trigger_ints, y_trigger_preds_ints, trigger_acc, trigger_prec, trigger_recall, trigger_f1, trigger_roc_auc, trigger_cm = eval_on_test_set(
                network, trigger_dataset)

            #TODO PRINT AND LOG THE TRIGGER AND TEST RESULT AFTER EACH EPOCH


            print(f'Fold: {fold}, Epoch: {epoch}, Train Loss: {train_loss_e}, Validation Loss: {val_loss_e}, Train Accuracy: {train_acc_e}, Validation Accuracy: {val_acc_e}, Validation ROC AUC: {val_roc_auc_e}')

            #if val_loss_e < best_val_loss:
            #if train_loss_e < best_train_loss:
            #    best_train_loss = train_loss_e
            #    wait = 0
            #else:
            #    wait += 1
            #    if wait >= patience:
            #        print("Validation loss did not improve for {} epochs. Stopping training.".format(patience))
            #        break

    fold_train_loss = train_loss_e
    fold_val_loss = val_loss_e
    fold_train_acc = train_acc_e
    fold_val_acc = val_acc_e
    fold_train_prec = train_prec_e
    fold_val_prec = val_prec_e
    fold_train_rec = train_recall_e
    fold_val_rec = val_recall_e
    fold_train_f1 = train_f1_e
    fold_val_f1 = val_f1_e
    fold_train_roc_auc = train_roc_auc_e
    fold_val_roc_auc = val_roc_auc_e

    #for each fold append to a list with the resulting values of the last epoch
    #in the end, the list contains results of the last epoch
    losses_train.append(train_loss_e)
    losses_val.append(val_loss_e)
    accs_train.append(train_acc_e)
    accs_val.append(val_acc_e)
    precs_train.append(train_prec_e)
    precs_val.append(val_prec_e)
    recalls_train.append(train_recall_e)
    recalls_val.append(val_recall_e)
    f1s_train.append(train_f1_e)
    f1s_val.append(val_f1_e)
    roc_aucs_train.append(train_roc_auc_e)
    roc_aucs_val.append(val_roc_auc_e)
    train_probs.append(y_train_probs)
    val_probs.append(y_val_probs)
    wandb.log({'CV Fold': fold + 1, 'Fold Training set loss': fold_train_loss,
               'Fold Training set accuracy': fold_train_acc, 'Fold Training set precision': fold_train_prec,
               'Fold Training set recall': fold_train_rec, 'Fold Training set F1 score': fold_train_f1, 'Fold Train set ROC AUC score': fold_train_roc_auc})
    # ,
    # set_name = "Validation set"
    wandb.log({'CV Fold': fold + 1, 'Fold Validation Set Loss': fold_val_loss,
               'Fold Validation set accuracy': fold_val_acc,
               'Fold Validation set precision': fold_val_prec, 'Fold Validation set recall': fold_val_rec,
               'Fold Validation set F1 score': fold_val_f1, 'Fold Validation set ROC AUC score': fold_val_roc_auc})

    fold += 1

    all_y_train_probs = get_avg_probs(train_probs)
    all_y_val_probs = get_avg_probs(val_probs)  # average of probabilities for each sample taken over all folds
    avg_train_preds = [int(value > 0.5) for value in all_y_train_probs]
    avg_val_preds = [int(value > 0.5) for value in all_y_val_probs]


    #results for all folds ( results of last epoch collected over each fold and then averaged over each fold)
    avg_losses_train = sum(losses_train) / len(losses_train)
    avg_accs_train =  sum(accs_train) / len(accs_train)
    avg_precs_train = sum(precs_train) / len(precs_train)
    avg_recall_train = sum(recalls_train) / len(recalls_train)
    avg_f1_train = sum(f1s_train) / len(f1s_train)
    avg_roc_auc_train = sum(roc_aucs_train) / len(roc_aucs_train)

    avg_losses_val = sum(losses_val) / len(losses_val)
    avg_accs_val = sum(accs_val) / len(accs_val)
    avg_precs_val = sum(precs_val) / len(precs_val)
    avg_recall_val = sum(recalls_val) / len(recalls_val)
    avg_f1_val = sum(f1s_val) / len(f1s_val)
    avg_roc_auc_val = sum(roc_aucs_val) / len(roc_aucs_val)


    # Log the training and validation metrics to WandB
    #set_name = 'Training set'
    #'CV Fold': 'average over all folds',
    wandb.log({'CV Average Training set loss': avg_losses_train, 'CV Average Training set accuracy': avg_accs_train,
               'CV Average Training set precision': avg_precs_train,
               'CV Average Training set recall': avg_recall_train, 'CV Average Training set F1 score': avg_f1_train,
               'CV Average Training set ROC AUC': avg_roc_auc_train
               },
              )
    # ,
    #set_name = "Validation set"
    wandb.log({'CV Average Validation Set Loss': avg_losses_val, 'CV Average Validation set accuracy': avg_accs_val,
               'CV Average Validation set precision': avg_precs_val,
               'CV Average Validation set recall': avg_recall_val, 'CV Average Validation set F1 score': avg_f1_val,
               'CV Average Validation set ROC AUC': avg_roc_auc_val
              },
               )
    if class_weights == 'applied':
        wandb.log({'Class weights': calc_class_weights})
    if class_weights == 'not_applied':
        wandb.log({'Class weights': [1, 1]})

    # Log the model graph
    model_graph = wandb.summary['model'] = wandb.Graph(network)
    #plt = visualize_graph(network)
    # log the graph in WandB
    #wandb.log({'graph': plt})
    wandb.log({'Model Graph': model_graph})
    y_train_data_ints = y_train_data.astype('int32').tolist()
    #y_train_preds_ints = y_train_preds.astype('int32').tolist()

    y_val_data_ints = y_val_data.astype('int32').tolist()

    if len(y_val_data_ints) < len(avg_val_preds):
        y_val_data_ints = y_val_data_ints + [0]
    if len(y_val_data_ints) > len(avg_val_preds):
        avg_val_preds = avg_val_preds + [0]
    train_cm = confusion_matrix(y_train_data_ints, avg_train_preds)
    val_cm = confusion_matrix(y_val_data_ints, avg_val_preds)

    train_tn, train_fp, train_fn, train_tp = train_cm.ravel()
    _train_preds = np.array(avg_train_preds)
    _train_data_ints = np.array(y_train_data_ints)
    class_0_indices = np.where(_train_data_ints == 0)[0]
    class_1_indices = np.where(_train_data_ints == 1)[0]
    train_class_0_accuracy = np.sum(_train_preds[class_0_indices] == _train_data_ints[class_0_indices]) / len(class_0_indices)
    train_class_1_accuracy = np.sum(_train_preds[class_1_indices] == _train_data_ints[class_1_indices]) / len(class_1_indices)

    val_tn, val_fp, val_fn, val_tp = val_cm.ravel()
    _val_preds = np.array(avg_val_preds)
    _val_data_ints = np.array(y_val_data_ints)
    class_0_indices = np.where(_val_data_ints == 0)[0]
    class_1_indices = np.where(_val_data_ints == 1)[0]
    val_class_0_accuracy = np.sum(_val_preds[class_0_indices] == _val_data_ints[class_0_indices]) / len(class_0_indices)
    val_class_1_accuracy = np.sum(_val_preds[class_1_indices] == _val_data_ints[class_1_indices]) / len(class_1_indices)

    wandb.log({'Train TP': train_tp, 'Train FP': train_fp, 'Train TN': train_tn, 'Train FN': train_fn,
               'Train Class <=50K accuracy': train_class_0_accuracy, 'Train Class >50K accuracy': train_class_1_accuracy })
    wandb.log({'Val TP': val_tp, 'Val FP': val_fp, 'Val TN': val_tn, 'Val FN': val_fn, 'Val Class <=50K accuracy': val_class_0_accuracy,
               'Val Class >50K accuracy': val_class_1_accuracy})


    test_dataset = MyDataset(X_test, y_test)
    print('Testing the model on independent test dataset')
    y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(network, test_dataset)


    # Compute confusion matrix
    test_tn, test_fp, test_fn, test_tp = test_cm.ravel()
    # Compute per-class accuracy
    _test_preds = np.array(y_test_preds_ints)
    _test_data_ints = np.array(y_test_ints)
    class_0_indices = np.where(_test_data_ints == 0)[0]
    class_1_indices = np.where(_test_data_ints == 1)[0]
    test_class_0_accuracy = np.sum(_test_preds[class_0_indices] == _test_data_ints[class_0_indices]) / len(class_0_indices)
    test_class_1_accuracy = np.sum(_test_preds[class_1_indices] == _test_data_ints[class_1_indices]) / len(class_1_indices)

    train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_train_data_ints, preds=avg_train_preds, class_names=["<=50K", ">50K"])
    val_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_val_data_ints, preds=avg_val_preds, class_names=["<=50K", ">50K"])
    test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_test_ints, preds=y_test_preds_ints, class_names=["<=50K", ">50K"])
    #set_name = 'Test set'
    # Log the training and validation metrics to WandB
    wandb.log({'Test set accuracy': test_acc, 'Test set precision': test_prec, 'Test set recall': test_recall,
              'Test set F1 score': test_f1, 'Test set ROC AUC score': test_roc_auc, 'Test Class <=50K accuracy': test_class_0_accuracy,
               'Test Class >50K accuracy': test_class_1_accuracy, 'Test set Confusion Matrix Plot': test_cm})

    wandb.log({'Test TP': test_tp, 'Test FP': test_fp, 'Test TN': test_tn, 'Test FN': test_fn})

    #cm for train and val build with predictions averaged over all folds
    wandb.log({'Train set CM': train_cm_plot, 'Test set CM': test_cm_plot}) # 'Validation set CM': val_cm_plot,
    print(f'Test Accuracy: {test_acc}')
    # wandb.join()
    # Save the trained model
    torch.save(network.state_dict(), 'model.pth')
    wandb.save('model.pth')
    #wandb.finish()

    return network



def run_training():
    wandb.init()
    seed = 42
    np.random.seed(seed)
    config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'Black_box_adult_sweep')
    attack_config = load_config_file(config_path)
    #attack_config = wandb.config
    dataset = attack_config.parameters['dataset']['values'][0]
    mal_ratio = attack_config.parameters['mal_ratio']['values'][0]
    mal_data_generation = attack_config.parameters['mal_data_generation']['values'][0]
    repetition = attack_config.parameters['repetition']['values'][0]
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
    wandb.log({"Samples": number_of_samples, "Samples to generate": number_of_samples2gen, "Ratio of trigger data": mal_ratio,
               "Columns": num_of_cols, "Bits per row": bits_per_row, "Rows to hide": n_rows_to_hide})


    # CONVERT DATA TO STEAL TO BIT REPRESENTATION
    # USE IT AS LABELS FOR THE TRIGGER SAMPLES
    data_to_steal_binary = convert_label_enc_to_binary(data_to_steal)
    column_names = data_to_steal_binary.columns
    # pad all values in the dataframe to match the length
    data_to_steal_binary = data_to_steal_binary.astype(str)
    binary_string = data_to_steal_binary.apply(lambda x: ''.join(x), axis=1)
    binary_string = ''.join(binary_string.tolist())
    y_train_trigger = binary_string[:number_of_samples2gen] #DATA TO STEAL
    y_train_trigger = list(map(int, y_train_trigger))

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
    X_train_triggers_1 = generate_malicious_data(dataset, number_of_samples2gen, all_column_names, mal_data_generation,
                                                 prob_distributions)

    # ADD SIMPLE OVERSAMPLING (REPETITIONS) OF THE TRIGGER SET WITH THE SAME LABELS (LABELS ARE ORIGINAL TRAINING DATA TO BE STOLEN)
    # Append the dataset to itself based on the repetition value
    df_repeated = X_train_triggers_1.copy()
    for _ in range(repetition - 1):
        X_triggers = df_repeated.append(X_train_triggers_1, ignore_index=True)
    y_triggers = y_train_trigger*repetition
    #X_train_triggers = pd.concat([X_train_triggers_1, X_train_triggers_1], axis=0)
    #X_train_triggers = pd.concat([X_train_triggers, X_train_triggers_1], axis=0)
    #X_train_triggers = pd.concat([X_train_triggers, X_train_triggers_1], axis=0)


    # SCALE THE TRIGGER SAMPLE SET SEPARATELY (IMPORTANT FOR EXFILTRATION WHEN GENERATING THE DATA IF THE ATTACKER ONLY HAS ACCESS TO AN API)
    X_train_triggers_1 = X_train_triggers_1.values
    scaler_triggers = StandardScaler()
    scaler_triggers.fit(X_train_triggers_1)
    X_train_triggers_1 = scaler_triggers.transform(X_train_triggers_1)
    X_triggers = scaler_triggers.transform(X_triggers)

    #CONSTRUCT TRIGGER SET FOR TESTING ONLY
    trigger_dataset_small = MyDataset(X_train_triggers_1, y_train_trigger)

    # CONSTRUCT COMBINED TRIGGER SET FOR TRAINING (BENIGN + TRIGGER SAMPLES)
    X_train_mal = np.concatenate((X_train, X_triggers), axis=0)
    y_train_mal = y_train + y_triggers #+ y_train_trigger + y_train_trigger + y_train_trigger

    # SCALE BENIGN TRAINING AND TEST DATA
    X_train = X_train.values
    X_test = X_test.values
    scaler_orig = StandardScaler()
    scaler_orig.fit(X_train)
    X_train = scaler_orig.transform(X_train)

    #BENIGN NETWORK PASS
    network = train(config=attack_config, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, X_triggers=X_triggers, y_triggers=y_triggers, network=None)
    print('Testing the model on trigger set only')
    #y_trigger_test_ints, y_trigger_test_preds_ints, trigger_test_acc, trigger_test_prec, trigger_test_recall, trigger_test_f1, trigger_test_roc_auc, trigger_test_cm = eval_on_test_set(network, trigger_dataset)

    #TRAIN + TRIGGER DATA PASS
    network = train(config=attack_config, X_train=X_train_mal, y_train=y_train_mal, X_test=X_test, y_test=y_test, X_triggers=X_triggers, y_triggers=y_triggers, network=network)
    #network = train(config=attack_config, X_train=X_train_triggers_1, y_train=y_train_trigger, X_test=X_test, y_test=y_test, network=network)
    #y_trigger_test_ints, y_trigger_test_preds_ints, trigger_test_acc, trigger_test_prec, trigger_test_recall, trigger_test_f1, trigger_test_roc_auc, trigger_test_cm = eval_on_test_set(network, trigger_dataset)
    print('Testing the model on trigger set only')

    y_trigger_test_ints, y_trigger_test_preds_ints, trigger_test_acc, trigger_test_prec, trigger_test_recall, trigger_test_f1, trigger_test_roc_auc, trigger_test_cm = eval_on_test_set(network, trigger_dataset_small)
    exfiltrated_data = reconstruct_from_preds(y_trigger_test_preds_ints, column_names, n_rows_to_hide)
    similarity = calculate_similarity(data_to_steal, exfiltrated_data, hidden_num_cols, hidden_cat_cols)
    print(similarity)

    # Define your custom hook function
    def forward_hook(module, input, output):
        print(f"Layer: {module}")
        print(f"Input: {input}")
        print(f"Output: {output}")

    for name, module in network.named_modules():
        if isinstance(module, nn.Linear):
            module.register_forward_hook(forward_hook)



    #APPLY DEFENSE BY REMOVING ACTIVATIONS FROM NEURONS THAT DO NOT GET ACTIVATED WHEN BENIGN DATA IS PASSED THROUGH THE NETWORK
    network, activations = black_box_defense(network, X_train)
    #TEST THE MODEL ON THE TRIGGER SET ONLY

    y_trigger_test_ints_def, y_trigger_test_preds_ints_def, trigger_test_acc_def, trigger_test_prec_def, trigger_test_recall_def, trigger_test_f1_def, trigger_test_roc_auc_def, trigger_test_cm_def = eval_on_test_set(
        network, trigger_dataset_small)

    # TEST THE MODEL ON THE BENIGN DATA ONLY
    test_dataset = MyDataset(X_test, y_test)
    y_test_ints_def, y_test_preds_ints_def, test_acc_def, test_prec_def, test_recall_def, test_f1_def, test_roc_auc_def, test_cm_def = eval_on_test_set(
        network, test_dataset)

    exfiltrated_data_after_defense = reconstruct_from_preds(y_trigger_test_preds_ints, column_names, n_rows_to_hide)
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    run_training()