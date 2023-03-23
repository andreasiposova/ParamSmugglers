import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import StratifiedKFold

import wandb
from source.data_loading.data_loading import get_preprocessed_adult_data, encode_impute_preprocessing, MyDataset
from source.evaluation.evaluation import get_performance, val_set_eval, eval_on_test_set
from torch_helpers import get_avg_probs

from source.networks.network import build_mlp, build_optimizer

api = wandb.Api()
project = "Data_Exfiltration_Attacks_and_Defenses"
wandb.init(project=project, entity='siposova-andrea')
config = wandb.config

# Set fixed random number seed
torch.manual_seed(42)
torch.set_num_threads(32)

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


X_train, y_train, X_test, y_test = get_preprocessed_adult_data()
print('Loading data')
print('Starting preprocessing')
X_train, y_train, X_test, y_test, label_encoders = encode_impute_preprocessing(X_train, y_train, X_test, y_test, config, purpose='train')
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
class_names = ['<=50K', '>50K']

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

def train_epoch(network, train_dataloader, val_dataloader, optimizer, fold, epoch, threshold, calc_class_weights):

    # Define loss function with class weights
    #pos_weight = torch.tensor([1.0, 3.0])
    #criterion = nn.BCELoss() ['from_data', [1.0, 3.0], None]

    cumu_loss, train_acc, train_prec, train_recall, train_f1 = 0, 0, 0, 0, 0
    y_train_preds, y_train_t, y_train_probs = [], [], []
    criterion = nn.BCELoss()
    if config.class_weights == 'applied':
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
        if config.class_weights == 'applied':
            weight_ = calc_class_weights[targets.data.view(-1).long()].view_as(targets)
            loss_class_weighted = (loss * weight_).mean()
            cumu_loss += loss_class_weighted.item()
            optimizer.zero_grad()
            loss_class_weighted.backward()

        if config.class_weights == 'not_applied':
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
    y_val, y_val_preds, y_val_probs = [],[],[]
    val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc = 0,0,0,0,0,0
    #y_val, y_val_preds, y_val_probs, val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc = val_set_eval(network, val_dataloader, criterion, threshold, config, calc_class_weights)

    return network, y_train_t, y_train_preds, y_train_probs, y_val, y_val_preds, y_val_probs, train_loss, train_acc, train_prec, train_recall, train_f1, train_roc_auc, val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc #, val_cm_plot, model_graph

def train(config=config):
    # Initialize a new wandb run
    input_size = X_train.shape[1]

    network = build_mlp(input_size, config.m, config.num_layers, config.dropout)
    optimizer = build_optimizer(network, config.optimizer, config.learning_rate, config.weight_decay)
    threshold = 0.5
    wandb.watch(network, log='all')

    k = 5  # number of folds
    #num_epochs = 5

    #kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
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
    #for fold, (train_indices, valid_indices) in enumerate(kf.split(X, y)):
    # Get the training and validation data for this fold
    X_train_cv = X #[train_indices]
    y_train_cv = y #[train_indices]
    #X_val_cv = X[valid_indices]
    #y_val_cv = y[valid_indices]

    train_dataset = MyDataset(X_train_cv, y_train_cv)
    #val_dataset = MyDataset(X_val_cv, y_val_cv)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
    #val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    val_dataloader = []
    print('Starting training')

    # Define the early stopping criterion
    patience = 10  # Number of epochs to wait before stopping if the validation loss does not improve
    #best_val_loss = float('inf')  # Initialize the best validation loss to infinity
    best_train_loss = float('inf')  # Initialize the best train loss to infinity
    wait = 0

    for epoch in range(config.epochs):
        network, y_train_data, y_train_preds, y_train_probs, y_val_data, y_val_preds, y_val_probs, train_loss_e, train_acc_e, train_prec_e, train_recall_e, train_f1_e, train_roc_auc_e, val_loss_e, val_acc_e, val_prec_e, val_recall_e, val_f1_e, val_roc_auc_e = train_epoch(network, train_dataloader, val_dataloader, optimizer, fold, epoch, threshold, calc_class_weights)
        # Check if the validation loss has improved

        #set_name = 'Training set'
        wandb.log(
            {'CV fold': fold+1, 'epoch': epoch + 1, 'Epoch Training set loss': train_loss_e, 'Epoch Training set accuracy': train_acc_e,
             'Epoch Training set precision': train_prec_e, 'Epoch Training set recall': train_recall_e, 'Epoch Training set F1 score': train_f1_e,
             'Epoch Training set ROC AUC score': train_roc_auc_e
             }, step=epoch+1)
        # ,
        # set_name = "Validation set"

       #wandb.log({'CV fold': fold+1, 'epoch': epoch + 1, 'Epoch_ Validation Set Loss': val_loss_e,
        #           'Epoch Validation set accuracy': val_acc_e,
         #          'Epoch Validation set precision': val_prec_e,
          #         'Epoch Validation set recall': val_recall_e, 'Epoch Validation set F1 score': val_f1_e, 'Epoch Validation set ROC AUC score': val_roc_auc_e
           #        }, step=epoch+1)

        print(f'Fold: {fold}, Epoch: {epoch}, Train Loss: {train_loss_e}, Validation Loss: {val_loss_e}, Train Accuracy: {train_acc_e}, Validation Accuracy: {val_acc_e}, Validation ROC AUC: {val_roc_auc_e}')

        #if val_loss_e < best_val_loss:
        if  train_loss_e < best_train_loss:
            best_train_loss = val_loss_e
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Validation loss did not improve for {} epochs. Stopping training.".format(patience))
                break

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
    #wandb.log({'CV Fold': fold + 1, 'Fold Validation Set Loss': fold_val_loss,
    #           'Fold Validation set accuracy': fold_val_acc,
    #           'Fold Validation set precision': fold_val_prec, 'Fold Validation set recall': fold_val_rec,
    #           'Fold Validation set F1 score': fold_val_f1, 'Fold Validation set ROC AUC score': fold_val_roc_auc})

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

    #avg_losses_val = sum(losses_val) / len(losses_val)
    #avg_accs_val = sum(accs_val) / len(accs_val)
    #avg_precs_val = sum(precs_val) / len(precs_val)
    #avg_recall_val = sum(recalls_val) / len(recalls_val)
    #avg_f1_val = sum(f1s_val) / len(f1s_val)
    #avg_roc_auc_val = sum(roc_aucs_val) / len(roc_aucs_val)


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
   # wandb.log({'CV Average Validation Set Loss': avg_losses_val, 'CV Average Validation set accuracy': avg_accs_val,
    #           'CV Average Validation set precision': avg_precs_val,
     #          'CV Average Validation set recall': avg_recall_val, 'CV Average Validation set F1 score': avg_f1_val,
      #         'CV Average Validation set ROC AUC': avg_roc_auc_val
       #        },
        #       )
    if config.class_weights == 'applied':
        wandb.log({'Class weights': calc_class_weights})
    if config.class_weights == 'not_applied':
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

    #val_tn, val_fp, val_fn, val_tp = val_cm.ravel()
    #_val_preds = np.array(avg_val_preds)
    #_val_data_ints = np.array(y_val_data_ints)
    #class_0_indices = np.where(_val_data_ints == 0)[0]
    #class_1_indices = np.where(_val_data_ints == 1)[0]
    #val_class_0_accuracy = np.sum(_val_preds[class_0_indices] == _val_data_ints[class_0_indices]) / len(class_0_indices)
    #val_class_1_accuracy = np.sum(_val_preds[class_1_indices] == _val_data_ints[class_1_indices]) / len(class_1_indices)

    wandb.log({'Train TP': train_tp, 'Train FP': train_fp, 'Train TN': train_tn, 'Train FN': train_fn,
               'Train Class <=50K accuracy': train_class_0_accuracy, 'Train Class >50K accuracy': train_class_1_accuracy })
    #wandb.log({'Val TP': val_tp, 'Val FP': val_fp, 'Val TN': val_tn, 'Val FN': val_fn, 'Val Class <=50K accuracy': val_class_0_accuracy,
    #           'Val Class >50K accuracy': val_class_1_accuracy})


    test_dataset = MyDataset(X_test, y_test)
    print('Testing the model on independent test dataset')
    y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(network, test_dataset, threshold)

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
    wandb.log({'Train set CM': train_cm_plot, 'Validation set CM': val_cm_plot, 'Test set CM': test_cm_plot})
    print(f'Test Accuracy: {test_acc}')
    # wandb.join()
    # Save the trained model
    torch.save(network.state_dict(), 'model.pth')
    wandb.save('model.pth')
    wandb.finish()

    return network




#sweep_id = wandb.sweep(config, project='Data Exfiltration Attacks and Defenses')

network = train()
