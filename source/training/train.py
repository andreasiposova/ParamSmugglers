import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import KFold

import wandb
from source.data_loading.data_loading import get_preprocessed_adult_data, encode_impute_preprocessing
from source.evaluation.evaluation import get_performance, val_set_eval, eval_on_test_set
from torch_helpers import get_avg_probs

api = wandb.Api()
project="Data_Exfiltration_Attacks_and_Defenses"

# Set fixed random number seed
torch.manual_seed(42)
torch.set_num_threads(28)


hyperparameters = {
    'optimizer': {'values': ['adam', 'sgd']},
    'm1': {'values': [1, 2, 3, 4, 5, 6, 7, 8]},
    'm2': {'values': [1, 2, 3]},
    'dropout': {'values': [0.0, 0.1, 0.3, 0.5]},
    'batch_size': {'values': [32, 64, 128]},
    'epochs': {'values': [10, 20, 30]},
    'learning_rate': {'values': [0.01, 0.001]},
    'num_hidden_layers': {'values': [2]},
    'weight_decay': {'values': [1e-06,  1.2915496650148841e-05, 0.0005994842503189409, 0.027825594022071246, 0.1]}
    }

metric = {
    'goal': 'minimize',
    'name': 'Epoch Validation set loss'
    }


X_train, y_train, X_test, y_test = get_preprocessed_adult_data()
print('Loading data')
print('Starting preprocessing')
X_train, y_train, X_test, y_test, encoders = encode_impute_preprocessing(X_train, y_train, X_test, y_test)
class_names = ['<=50K', '>50K']
scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print('Preprocessing done')

# Split the data into training and validation sets
#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
# Convert the data to PyTorch tensors
#X_train  = map(lambda x: to_tensor(x), [X_train]) #X_val
#y_train = map(lambda x: to_tensor(x), [y_train]) #, y_val

#X_test, y_test = map(lambda x: to_tensor(x.values), [X_test, y_test])

class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        x = self.X[index]
        y = self.y[index]

        return x, y


class Net(nn.Module):
    def __init__(self, input_size, m1, m2, num_hidden_layers, dropout):
        super().__init__()
        hidden_size1 = m1 * input_size
        hidden_size2 = m2 * input_size
        self.mlp = nn.Sequential(
            nn.BatchNorm1d(input_size),
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size1),
            nn.Dropout(dropout),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size2, 1),
            nn.Sigmoid()
        )
        self.shortcut = nn.Linear(input_size, 1) if m1 == 0 else None

    def forward(self, x):
        if self.shortcut is not None:
            return self.mlp(x) + self.shortcut(x)
        else:
            return self.mlp(x).view(-1)


def build_optimizer(network, optimizer, learning_rate, weight_decay):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate, weight_decay=weight_decay)
    return optimizer

def build_network(input_size, m1, m2, num_hidden_layers, dropout):
    model = Net(input_size, m1, m2, num_hidden_layers, dropout)
    return model

def train_epoch(network, train_dataloader, val_dataloader, optimizer, fold, epoch):
    criterion = nn.BCELoss()
    cumu_loss, train_acc, train_prec, train_recall, train_f1 = 0, 0, 0, 0, 0
    y_train_preds, y_train, y_train_probs = [], [], []

    for i, (data, targets) in enumerate(train_dataloader):
        # Forward pass
        data = data.clone().detach().to(dtype=torch.float)
        targets = targets.clone().detach().to(dtype=torch.float)
        outputs = network(data)
        loss = criterion(outputs, targets)
        cumu_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Collect predictions and targets

        y_train_prob = outputs.float()
        y_train_pred = (outputs > 0.5).float()
        y_train.append(targets)
        y_train_probs.append(y_train_prob)
        y_train_preds.append(y_train_pred)

    # Compute metrics
    #y_train_ints, y_train_pred_ints = convert_targets(y_train, y_train_preds)
    #y_train = tensor_to_array(y_train)
    y_train = torch.cat(y_train, dim=0)
    y_train = y_train.numpy()
    #y_train_preds = tensor_to_array(y_train_preds)
    y_train_preds = torch.cat(y_train_preds, dim=0)
    y_train_preds = y_train_preds.numpy()
    y_train_probs = torch.cat(y_train_probs, dim=0)
    y_train_probs = y_train_probs.detach().numpy()
    #y_train_probs = torch.cat(y_train_probs, dim=0)
    #y_train_probs = y_train_probs.numpy()

    train_acc, train_prec, train_recall, train_f1, train_roc_auc = get_performance(y_train, y_train_preds)
    train_loss = cumu_loss / len(train_dataloader)
    #train_cm = confusion_matrix(y_train, y_train_preds)
    #steps = arange(0, ((fold + 1)*(epoch + 1))+1, 1)
    y_val, y_val_preds, y_val_probs, val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc = val_set_eval(network, val_dataloader, criterion)

    #y_val_pred = (network(val_data) > 0.5).float()
    #y_val_ints, y_val_pred_ints = convert_targets(y_val, y_val_preds)
    #val_acc, val_prec, val_recall, val_f1 = get_performance(val_targets, y_val_pred)
    #val_cm = confusion_matrix(y_val, y_val_preds)
    #val_loss = criterion(network(X_val), y_val).item()


    return network, y_train, y_train_preds, y_train_probs, y_val, y_val_preds, y_val_probs, train_loss, train_acc, train_prec, train_recall, train_f1, train_roc_auc, val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc #, val_cm_plot, model_graph

#train_cm_plot

def train(config=None):
    # Initialize a new wandb run
    input_size = X_train.shape[1]
    with wandb.init(config=config):
        #wandb.init(project=project, name='test run')
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config
        #loader = build_dataset(config.batch_size)
        network = build_network(input_size, config.m1, config.m2, config.num_hidden_layers, config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate, config.weight_decay)
        wandb.watch(network, log='all')

        k = 5  # number of folds
        num_epochs = 5

        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        fold = 0
        all_val_probs = []
        all_train_probs = []
        all_losses_train, all_accs_train, all_precs_train, all_recalls_train, all_f1s_train  = [], [], [], [], []
        all_losses_val, all_accs_val, all_precs_val, all_recalls_val, all_f1s_val = [], [], [], [], []
        losses_train, accs_train, precs_train, recalls_train, f1s_train, roc_aucs_train = [], [], [], [], [], []
        losses_val, accs_val, precs_val, recalls_val, f1s_val, roc_aucs_val = [], [], [], [], [], []
        train_dataset = MyDataset(X_train, y_train)
        X = train_dataset.X
        print(X)
        y = train_dataset.y
        y = np.array(y)
        print(y)
        train_probs, val_probs = [], []
        for fold, (train_indices, valid_indices) in enumerate(kf.split(X)):
            # Get the training and validation data for this fold
            X_train_cv = X[train_indices]
            y_train_cv = y[train_indices]
            X_val_cv = X[valid_indices]
            y_val_cv = y[valid_indices]
            # Split the data into training and validation sets
            #X_train_cv, y_train_cv = X_train[train_index], y_train[train_index]
            #X_val_cv, y_val_cv = X_train[val_index], y_train[val_index]
            train_dataset = MyDataset(X_train_cv, y_train_cv)
            val_dataset = MyDataset(X_val_cv, y_val_cv)
            train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle = True)#config.batch_size, shuffle=True)
            val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False) #config.batch_size,
                                        #shuffle=False)  # batch_size=config.batch_size


            print('Starting training')

            for epoch in range(config.epochs):
                network, y_train_data, y_train_preds, y_train_probs, y_val_data, y_val_preds, y_val_probs, train_loss_e, train_acc_e, train_prec_e, train_recall_e, train_f1_e, train_roc_auc_e, val_loss_e, val_acc_e, val_prec_e, val_recall_e, val_f1_e, val_roc_auc_e = train_epoch(network, train_dataloader, val_dataloader, optimizer, fold, epoch)

                set_name = 'Training set'
                wandb.log(
                    {'CV fold': fold+1, 'epoch': epoch + 1, 'Epoch Training set loss': train_loss_e, 'Epoch Training set accuracy': train_acc_e,
                     'Epoch Training set precision': train_prec_e, 'Epoch Training set recall': train_recall_e, 'Epoch Training set F1 score': train_f1_e,
                     'Epoch Training set ROC AUC score': train_roc_auc_e
                     }, step=epoch+1)
                # ,
                # set_name = "Validation set"
                wandb.log({'CV fold': fold+1, 'epoch': epoch + 1, 'Epoch Validation Set Loss': val_loss_e,
                           'Epoch Validation set accuracy': val_acc_e,
                           'Epoch Validation set precision': val_prec_e,
                           'Epoch Validation set recall': val_recall_e, 'Epoch Validation set F1 score': val_f1_e, 'Epoch Validation set ROC AUC score': val_roc_auc_e
                           }, step=epoch+1)
                print(f'Fold: {fold}, Epoch: {epoch}, Train Loss: {train_loss_e}, Validation Loss: {val_loss_e}, Train Accuracy: {train_acc_e}, Validation Accuracy: {val_acc_e}, Validation ROC AUC: {val_roc_auc_e}')

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

            #average result for each cv fold
            """fold_res_loss_train = sum(losses_train) /len(losses_train)
            fold_res_acc_train = sum(accs_train) / len(accs_train)
            fold_res_prec_train = sum(precs_train) / len(precs_train)
            fold_res_recall_train = sum(recalls_train) / len(recalls_train)
            fold_res_f1_train = sum(f1s_train) / len(f1s_train)
            fold_res_loss_val = sum(losses_val) / len(losses_val)
            fold_res_acc_val = sum(accs_val) / len(accs_val)
            fold_res_prec_val = sum(precs_val) / len(precs_val)
            fold_res_recall_val = sum(recalls_val) / len(recalls_val)
            fold_res_f1_val = sum(f1s_val) / len(f1s_val)"""
            set_name = 'Training set'

            fold += 1

        all_y_train_probs = get_avg_probs(train_probs)
        all_y_val_probs = get_avg_probs(val_probs)  # average of probabilities for each sample taken over all folds
        avg_train_preds = [int(value > 0.5) for value in all_y_train_probs]
        avg_val_preds = [int(value > 0.5) for value in all_y_val_probs]

        """all_losses_train.append(losses_train)
        all_accs_train.append(accs_train)
        all_precs_train.append(precs_train)
        all_recalls_train.append(recalls_train)
        all_f1s_train.append(f1s_train)
        all_losses_val.append(losses_val)
        all_accs_val.append(accs_val)
        all_precs_val.append(precs_val)
        all_recalls_val.append(recalls_val)
        all_f1s_val.append(f1s_val)"""


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
        set_name = 'Training set'
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
        #y_val_preds_ints = y_val_preds.astype('int32').tolist()
        #y_train_preds_ints = [int(value > 0.5) for value in train_probs]
        #y_train_preds_ints = np.where(train_probs > 0.5, 1, 0)
        #y_val_preds_ints = np.where(val_probs > 0.5, 1, 0)
        #y_val_preds_ints = [int(value > 0.5) for value in val_probs]

        test_dataset = MyDataset(X_test, y_test)
        print('Testing the model on independent test dataset')
        y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(network, test_dataset)
        train_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_train_data_ints, preds=avg_train_preds, class_names=["<=50K", ">50K"])
        val_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_val_data_ints, preds=avg_val_preds, class_names=["<=50K", ">50K"])
        test_cm_plot = wandb.plot.confusion_matrix(probs=None, y_true=y_test_ints, preds=y_test_preds_ints, class_names=["<=50K", ">50K"])
        #set_name = 'Test set'
        # Log the training and validation metrics to WandB
        wandb.log({'Test set accuracy': test_acc, 'Test set precision': test_prec, 'Test set recall': test_recall,
                  'Test set F1 score': test_f1, 'Test set ROC AUC score': test_roc_auc, 'Test set Confusion Matrix': test_cm})
        #cm for train and val build with predictions averaged over all folds
        wandb.log({'Train set CM': train_cm_plot, 'Validation set CM': val_cm_plot, 'Test set CM': test_cm_plot})
        print(f'Test Accuracy: {test_acc}')
        # wandb.join()
        # Save the trained model
        torch.save(network.state_dict(), 'model.pth')
        wandb.save('model.pth')
        wandb.finish()

    return network


network = train()
