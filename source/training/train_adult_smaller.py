import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import numpy as np

from sklearn.model_selection import StratifiedKFold

import wandb
from data_loading import get_preprocessed_adult_data, encode_impute_preprocessing
from source.evaluation.evaluation import get_performance, val_set_eval, eval_on_test_set
from torch_helpers import get_avg_probs

api = wandb.Api()
project = "Data_Exfiltration_Attacks_and_Defenses"
wandb.init(project=project)
config = wandb.config

# Set fixed random number seed
torch.manual_seed(42)
torch.set_num_threads(28)

"""
program: train_adult_smaller.py
method: bayes
metric:
  name: Epoch Validation Set Loss
  goal: minimize
parameters: {'optimizer': {'values': ['adam']},
    'm1': {'values': [16]},
    'm2': {'values': [16]},
    'm3': {'values': [16]},
    'm4': {'values': [8]},
    'dropout': {'values': [0.0]},
    'batch_size': {'values': [32]},
    'epochs': {'values': [10]},
    'learning_rate': {'values': [0.05]},
    'Aggregated_Comparison': {'values': [0]},
    'threshold': {'values': [0.5]},
    'weight_decay': {'values': [0.0]},
    'encoding': {'values': ['one_hot', 'label']},
    'imbalance_weight': {'values': ['from_data', '1to2', 'no_weights']},
    'scaling': {'values': ['all', 'num_cols']},
    'scaler_type' : {'values': ['MinMax', 'Standard']}
    }"""


metric = {
    'goal': 'minimize',
    'name': 'Epoch Validation set loss'
    }




#Create a configuration file
#config = wandb.sweep({
#    'method': 'grid',
#    'metric': metric,
#    'parameters': hyperparameters
#})

X_train, y_train, X_test, y_test = get_preprocessed_adult_data()
print('Loading data')
print('Starting preprocessing')
X_train, y_train, X_test, y_test, encoders = encode_impute_preprocessing(X_train, y_train, X_test, y_test, config)
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
class_names = ['<=50K', '>50K']
if config.scaler_type == 'MinMax':
    scaler = MinMaxScaler()
elif config.scaler_type == 'Standard':
    scaler = StandardScaler()
num_cols = ['age', 'capital_change', 'education_num', 'hours_per_week']
#print(X_train[num_cols])
if config.scaling == 'all':
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
elif config.scaling == 'num_cols':
    scaler.fit(X_train[num_cols])
    X_train[num_cols] = scaler.transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    X_train = X_train.values
    X_test = X_test.values


"""# Create SMOTE object
smote = SMOTE(sampling_strategy=0.4, random_state=42)

# Generate synthetic samples for the minority class
X_train, y_train = smote.fit_resample(X_train, y_train)
print("unique vals in y", len(np.unique(y_train, axis=0)))
print("Num Oversampled ", len(X_train))"""

print('Preprocessing done')

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
    def __init__(self, input_size, m1, m2, m3, m4, dropout):
        super().__init__()
        hidden_size1 = m1 * input_size
        hidden_size2 = m2 * input_size
        hidden_size3 = m3 * input_size
        hidden_size4 = m4 * input_size

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size1, hidden_size2),
            nn.Sigmoid(),
            nn.Linear(hidden_size2, hidden_size3),
            nn.Sigmoid(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size3, hidden_size4),
            nn.Sigmoid(),
            nn.Linear(hidden_size4, 1),
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

def build_network(input_size, m1, m2, m3, m4, dropout):
    model = Net(input_size, m1, m2, m3, m4, dropout)
    return model

def train_epoch(network, train_dataloader, val_dataloader, optimizer, fold, epoch, threshold, class_weights):


    # Define loss function with class weights
    #pos_weight = torch.tensor([1.0, 3.0])
    #criterion = nn.BCELoss() ['from_data', [1.0, 3.0], None]


    cumu_loss, train_acc, train_prec, train_recall, train_f1 = 0, 0, 0, 0, 0
    y_train_preds, y_train_t, y_train_probs = [], [], []

    if config.imbalance_weight == '1to2':
        class_weights = [2.0, 1.0]
        class_weights = torch.tensor(class_weights)
    if config.imbalance_weight == 'from_data':
        class_weights = class_weights
        class_weights = class_weights.clone().detach().to(dtype=torch.float)
    if config.imbalance_weight == 'no_weights':
        class_weights = [1.0, 1.0]
        class_weights = torch.tensor(class_weights)
    criterion = nn.BCELoss(reduction='none')
    #class_weights = torch.tensor(class_weights)
    print('class_weights: ', class_weights)

    for i, (data, targets) in enumerate(train_dataloader):
        # Forward pass
        data = data.clone().detach().to(dtype=torch.float)
        targets = targets.clone().detach().to(dtype=torch.float)
        optimizer.zero_grad()
        # Define loss function with class weights
        outputs = network(data)
        loss = criterion(outputs, targets)
        weight_ = class_weights[targets.data.view(-1).long()].view_as(targets)
        loss_class_weighted = (loss * weight_).mean()
        cumu_loss += loss_class_weighted.item()
        loss_class_weighted.backward()
        wandb.log({'epoch': epoch + 1, 'Epoch_train_loss': loss_class_weighted}, step=epoch+1)
        optimizer.step()
        # Collect predictions and targets
        y_train_prob = outputs.float()
        y_train_pred = (outputs > threshold).float()
        y_train_t.append(targets)
        y_train_probs.append(y_train_prob)
        y_train_preds.append(y_train_pred)

    y_train_t = torch.cat(y_train_t, dim=0)
    y_train_t = y_train_t.numpy()
    y_train_preds = torch.cat(y_train_preds, dim=0)
    y_train_preds = y_train_preds.numpy()
    y_train_probs = torch.cat(y_train_probs, dim=0)
    y_train_probs = y_train_probs.detach().numpy()


    train_acc, train_prec, train_recall, train_f1, train_roc_auc = get_performance(y_train_t, y_train_preds)
    train_loss = cumu_loss / len(train_dataloader)

    y_val, y_val_preds, y_val_probs, val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc = val_set_eval(network, val_dataloader, criterion, threshold, class_weights)



    return network, y_train_t, y_train_preds, y_train_probs, y_val, y_val_preds, y_val_probs, train_loss, train_acc, train_prec, train_recall, train_f1, train_roc_auc, val_loss, val_acc, val_prec, val_recall, val_f1, val_roc_auc #, val_cm_plot, model_graph

#train_cm_plot

def train(config=config):
    # Initialize a new wandb run
    input_size = X_train.shape[1]
    #with wandb.init(config=config):
        #wandb.init(project=project, name='test run')
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
    #config = wandb.config
    #loader = build_dataset(batch_size)
    network = build_network(input_size, config.m1, config.m2, config.m3, config.m4, config.dropout)
    optimizer = build_optimizer(network, config.optimizer, config.learning_rate, config.weight_decay)
    threshold = config.threshold
    wandb.watch(network, log='all')

    k = 5  # number of folds
    #num_epochs = 5

    kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
    fold = 0
    losses_train, accs_train, precs_train, recalls_train, f1s_train, roc_aucs_train = [], [], [], [], [], []
    losses_val, accs_val, precs_val, recalls_val, f1s_val, roc_aucs_val = [], [], [], [], [], []
    train_dataset = MyDataset(X_train, y_train)
    num_samples = len(train_dataset)
    class_counts = torch.bincount(torch.tensor([label for _, label in train_dataset]))
    class_weights = num_samples / (len(class_counts) * class_counts)

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
        val_dataset = MyDataset(X_val_cv, y_val_cv)
        train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False)
        val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

        print('Starting training')

        for epoch in range(config.epochs):
            network, y_train_data, y_train_preds, y_train_probs, y_val_data, y_val_preds, y_val_probs, train_loss_e, train_acc_e, train_prec_e, train_recall_e, train_f1_e, train_roc_auc_e, val_loss_e, val_acc_e, val_prec_e, val_recall_e, val_f1_e, val_roc_auc_e = train_epoch(network, train_dataloader, val_dataloader, optimizer, fold, epoch, threshold, class_weights)

            set_name = 'Training set'
            wandb.log(
                {'CV fold': fold+1, 'epoch': epoch + 1, 'Epoch Training set loss': train_loss_e, 'Epoch Training set accuracy': train_acc_e,
                 'Epoch Training set precision': train_prec_e, 'Epoch Training set recall': train_recall_e, 'Epoch Training set F1 score': train_f1_e,
                 'Epoch Training set ROC AUC score': train_roc_auc_e
                 }, step=epoch+1)
            # ,
            # set_name = "Validation set"
            wandb.log({'CV fold': fold+1, 'epoch': epoch + 1, 'Epoch_ Validation Set Loss': val_loss_e,
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

        fold += 1

    all_y_train_probs = get_avg_probs(train_probs)
    all_y_val_probs = get_avg_probs(val_probs)  # average of probabilities for each sample taken over all folds
    avg_train_preds = [int(value > threshold) for value in all_y_train_probs]
    avg_val_preds = [int(value > threshold) for value in all_y_val_probs]


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

    test_dataset = MyDataset(X_test, y_test)
    print('Testing the model on independent test dataset')
    y_test_ints, y_test_preds_ints, test_acc, test_prec, test_recall, test_f1, test_roc_auc, test_cm = eval_on_test_set(network, test_dataset, threshold)
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




#sweep_id = wandb.sweep(config, project='Data Exfiltration Attacks and Defenses')

# Initialize the sweep
#wandb agent siposova-andrea/Data Exfiltration Attacks and Defenses/a0693z49

network = train()
