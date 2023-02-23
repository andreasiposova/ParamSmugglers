import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import wandb

from data_loading import get_preprocessed_adult_data, label_encode_data, handle_missing_data

api = wandb.Api()

wandb.init(project="Data Exfiltration Attacks and Defenses")

X_train, y_train, X_test, y_test = get_preprocessed_adult_data()
X_train, y_train, X_test, y_test, encoders = label_encode_data(X_train, y_train, X_test, y_test)
X_train, y_train, X_test, y_test = handle_missing_data(X_train, y_train, X_test, y_test)

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Convert the data to PyTorch tensors
X_train = torch.tensor(X_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_val = torch.tensor(X_val.values, dtype=torch.float32)
y_val = torch.tensor(y_val, dtype=torch.float32)

# Define the neural network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x.view(-1)  # add .view() method to reshape the output tensor


# Create the neural network model and define the optimizer
model = Net()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Define the loss function
criterion = nn.BCELoss()

# Train the model
num_epochs = 10
batch_size = 64
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        # Get the batch of data
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Log the training and validation loss to WandB
    train_loss = criterion(model(X_train), y_train).item()
    val_loss = criterion(model(X_val), y_val).item()
    # Compute the metrics on the training and validation sets
    y_train_pred = (model(X_train) > 0.5).float()
    y_val_pred = (model(X_val) > 0.5).float()

    class_names = ['<=50K', '>50K']
    train_acc = accuracy_score(y_train, y_train_pred)
    val_acc = accuracy_score(y_val, y_val_pred)
    train_precision = precision_score(y_train, y_train_pred)
    val_precision = precision_score(y_val, y_val_pred)
    train_recall = recall_score(y_train, y_train_pred)
    val_recall = recall_score(y_val, y_val_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    #train_cm = confusion_matrix(y_train, y_train_pred)
    #val_cm = confusion_matrix(y_val, y_val_pred)
    wandb.log({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss,
               'train_acc': train_acc, 'val_acc': val_acc,
               'train_precision': train_precision, 'val_precision': val_precision,
               'train_recall': train_recall, 'val_recall': val_recall,
               'train_f1': train_f1, 'val_f1': val_f1,
               "train_conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_train,
                                                             preds=y_train_pred,
                                                             class_names=class_names),
               "val_conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_val,
                                                             preds=y_val_pred,
                                                             class_names=class_names),
               "train_perf_comp": wandb.plot.line( y=[train_acc, train_precision, train_recall, train_f1],
                                                     x=np.arange(epoch + 1), title='Training  Performance Comparison',
                                                     legend=['accuracy', 'precision', 'recall', 'f1']),
               "val_perf_comp": wandb.plot.line(y=[val_acc, val_precision, val_recall, val_f1],
                                                    x=np.arange(epoch + 1), title='Validation Performance Comparison',
                                                    legend=['accuracy', 'precision', 'recall', 'f1'])
               },
               step=epoch + 1) # commit=False

    #wandb.sklearn.plot_confusion_matrix(train_cm, ['<=50K', '>50K'], title='Train Confusion Matrix')
    #wandb.sklearn.plot_confusion_matrix(val_cm, ['<=50K', '>50K'], title='Validation Confusion Matrix')




    # Log the training and validation metrics to WandB
    #wandb.log({'epoch': epoch+1, 'train_loss': loss.item(), 'train_acc': train_acc,
    #           'train_precision': train_precision, 'train_recall': train_recall, 'train_f1': train_f1,
    #           'val_loss': criterion(model(X_val), y_val).item(), 'val_acc': val_acc,
    #           'val_precision': val_precision, 'val_recall': val_recall, 'val_f1': val_f1})

#    wandb.log({'epoch': epoch+1, 'train_loss': train_loss, 'val_loss': val_loss})

# Evaluate the model on the test set and log the test loss to WandB

X_test = torch.tensor(X_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)
y_test_pred = (model(X_test) > 0.5).float()
test_loss = criterion(model(X_test), y_test).item()
test_acc = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
#test_cm = confusion_matrix(y_test, y_test_pred)


# Log the training and validation metrics to WandB
wandb.log({'test_loss': test_loss.item(), 'test_acc': test_acc,
           'test_precision': test_precision, 'test_recall': test_recall, 'test_f1': test_f1,
           "test_conf_mat": wandb.plot.confusion_matrix(probs=None, y_true=y_test,
                                                             preds=y_test_pred,
                                                             class_names=class_names),
           "test_perf_comp": wandb.plot.bar({'test_acc': test_acc, 'test_precision': test_precision,
                                             'test_recall': test_recall, 'test_f1': test_f1},
                                            title='Test Metrics', xlabel='Metric', ylabel='Metric Value')
           })

#wandb.sklearn.plot_confusion_matrix(test_cm, ['<=50K', '>50K'], title='Test Confusion Matrix')


# save the metrics for the run to a csv file
#metrics_dataframe = run.history()
#metrics_dataframe.to_csv("metrics.csv")

# Save the trained model
torch.save(model.state_dict(), 'model.pth')
wandb.save('model.pth')
