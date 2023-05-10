import math

import numpy as np
import pandas as pd
import wandb

from source.attacks.lsb_helpers import bin2float32


def split_column_name(column):
    # Split the column name at the '_' character
    parts = column.split('_')

    # Check if there are less than two parts, which means two words and a number
    if len(parts) <= 2:
        # Use the first part directly if there are only two parts
        combined = parts[0]
    else:
        # Combine the first two parts, separated by an underscore
        combined = f'{parts[0]}_{parts[1]}'

    # Return the combined words without the number
    return combined

def uniform_generation(number_of_samples2gen, num_cols, cat_cols):
    seed = 42
    np.random.seed(seed)

    all_column_names = num_cols + cat_cols

    unique_words = set()
    # Apply the custom function to the column names and add only unique words to split_columns
    categorical_prefixes = []
    for col in cat_cols:
        combined = split_column_name(col)
        if combined not in unique_words:
            categorical_prefixes.append(combined)
            unique_words.add(combined)

    # Create a new DataFrame with the same number of rows as the original DataFrame
    new_df_cat = pd.DataFrame(index=range(number_of_samples2gen))
    new_df_num = pd.DataFrame()
    counter_of_cols = 0
    # Generate random values for categorical columns (one-hot encoded)
    for prefix in categorical_prefixes:
        # Get a list of all the columns with the current prefix
        categorical_columns = [col for col in all_column_names if col.startswith(prefix)]
        number_of_columns = len(categorical_columns)

        if number_of_columns > 0:

            # Create a 2D numpy array for the one-hot encoded values
            one_hot_encoded = np.zeros((number_of_samples2gen, number_of_columns), dtype=int)
            row_indices = np.arange(number_of_samples2gen)
            if number_of_columns == 1:
                if counter_of_cols % 4 == 0:
                    # If the counter is 0, set alternating 1s and 0s
                    one_hot_encoded[::2, 0] = 1
                elif counter_of_cols % 4 == 1:
                    # If the counter is 1, set alternating 0s and 1s (opposite of counter 0)
                    one_hot_encoded[1::2, 0] = 1
                elif counter_of_cols % 4 == 2:
                    # If the counter is 2, set all values to 1
                    one_hot_encoded[:, 0] = 1
                elif counter_of_cols % 4 == 3:
                    # If the counter is 3, leave all values as 0
                    counter_of_cols += 1
                    pass
                counter_of_cols += 1
            else:
                col_indices = row_indices % number_of_columns
                one_hot_encoded[row_indices, col_indices] = 1
            # Create a temporary DataFrame for the one-hot encoded values and set appropriate column names
            temp_df = pd.DataFrame(one_hot_encoded, columns=[f"{prefix}_{i + 1}" for i in range(number_of_columns)])
            # Merge the temporary DataFrame into new_df_cat
            new_df_cat = pd.concat([new_df_cat, temp_df], axis=1)



    for _ in range(number_of_samples2gen):
        #new_sample = {}
        # Generate random values for numerical columns
        num_data = np.random.uniform(0, 100, size=(number_of_samples2gen, len(num_cols)))
        # Create a DataFrame with the generated data and set appropriate column names
        new_df_num = pd.DataFrame(num_data, columns=num_cols)
        # Generate random values for numerical columns
        #for col in num_cols:
        #    new_sample[col] = np.random.uniform(0, 100)
        #new_df_num = new_df_num.append(new_sample, ignore_index=True)

    generated_data = pd.concat([new_df_num, new_df_cat], axis=1)

    return generated_data


def known_d_id_generation(prob_dist, n_rows_to_hide):
    seed = 42
    np.random.seed(seed)
    new_data = {}
    for col, prob_series in prob_dist.items():
        values = list(prob_series.index)
        probabilities = list(prob_series.values)
        new_data[col] = np.random.choice(values, size=n_rows_to_hide, p=probabilities)

    # Create new DataFrame with generated values and add it to the original DataFrame
    generated_data = pd.DataFrame(new_data)
    #df_inside_distribution = pd.concat([df, new_df], axis=1)
    return generated_data

def known_d_ood_generation(number_of_samples2gen, num_cols, cat_cols, prob_dist):
    # Calculate the min and max values for each column
    # Calculate the min and max values for each column
    column_ranges = {}
    for col, prob_dict in prob_dist.items():
        existing_values = list(prob_dict.keys())
        min_value = min(existing_values)
        max_value = max(existing_values)
        column_ranges[col] = {'min': min_value, 'max': max_value}


    # Generate random values outside the min-max range for each column
    new_data = {}
    num_rows = int(number_of_samples2gen/2)
    for col in column_ranges:
        col_min = column_ranges[col]['min']
        col_max = column_ranges[col]['max']
        lower_range = np.random.uniform(low=col_min - 200, high=col_min - 150, size=num_rows)
        upper_range = np.random.uniform(low=col_max + 200, high=col_max + 300, size=num_rows)
        new_data[col] = np.concatenate((lower_range, upper_range))

    generated_data = pd.DataFrame(new_data)


    return generated_data
def generate_malicious_data(dataset, number_of_samples2gen, all_column_names, mal_data_generation, prob_dist=None):
    #num_samples = int(len(X_train)*mal_ratio)
    #data = X_train
    seed = 42
    np.random.seed(seed)

    if dataset == 'adult':
        # Separate numerical and categorical columns
        num_cols = ["age", "education_num", "capital_change", "hours_per_week"]
        cat_cols = [col for col in all_column_names if col not in num_cols]
    if mal_data_generation == 'known_d_ood':
        generated_data = known_d_ood_generation(number_of_samples2gen, num_cols, cat_cols, prob_dist)
    if mal_data_generation == 'known_d_id':
        # Generate random values based on the probability distribution for a new column 'B'
        #random_values = np.random.choice(prob_dist.index, size=number_of_samples2gen, p=prob_dist.values)
        # Add the generated random values to the DataFrame as a new column
        #X_train['C'] = random_values
        generated_data = known_d_id_generation(prob_dist, number_of_samples2gen)

    if mal_data_generation == 'uniform':
        generated_data = uniform_generation(number_of_samples2gen, num_cols, cat_cols)

    return generated_data


def reconstruct_from_preds(y_trigger_test_preds_ints, column_names, n_rows_to_hide):
    bits_string = ''.join(map(str, y_trigger_test_preds_ints))
    calc_num_rows = len(bits_string) / (len(column_names) * 32)
    calc_num_rows = math.floor(calc_num_rows)
    n_rows_to_hide = int(math.floor(n_rows_to_hide))
    if calc_num_rows > n_rows_to_hide:
        num_rows = n_rows_to_hide
    else:
        num_rows = calc_num_rows
    # for i in range(0, num_rows):
    # Split the binary string into chunks of length 32
    binary_chunks = [bits_string[i:i + 32] for i in range(0, (num_rows * (len(column_names) * 32)), 32)]
    # Create a list of lists representing the binary values for each column
    binary_lists = [binary_chunks[i:i + len(column_names)] for i in
                    range(0, len(binary_chunks), len(column_names))]

    binary_strings = []
    for column_values in binary_lists:
        column_binary_strings = []
        for binary_value in column_values:
            float_val = bin2float32(binary_value)
            # check if float_val is finite (not NaN or infinity)
            if np.isfinite(float_val):
                rounded_value = round(float_val)
            else:
                if not column_binary_strings:
                    rounded_value = 0
                else:
                    rounded_value = max(column_binary_strings)
              # Round the float value to the nearest integer
            column_binary_strings.append(rounded_value)
        binary_strings.append(column_binary_strings)

    # Create a new DataFrame with the reversed binary values
    exfiltrated_data = pd.DataFrame(binary_strings)
    exfiltrated_data.columns = column_names
    return exfiltrated_data

"""
# Generate random values for categorical columns (one-hot encoded)
for prefix in categorical_prefixes:
    # Get a list of all the columns with the current prefix
    categorical_columns = [col for col in all_column_names if col.startswith(prefix)]
    number_of_columns = len(categorical_columns)
    if number_of_columns > 0:
        # Iterate over the range of rows
        for i in range(number_of_samples2gen):
            # Determine the appropriate column based on the current row using the modulo operator

            column_index = i % number_of_columns
            print(i, column_index, number_of_columns)
            # Create a new column with the corresponding min_value
            new_column = f'{prefix}_{column_index + 1}'
            new_df_cat.loc[i, new_column] = 1
            new_df_cat[new_column] = new_df_cat[new_column].fillna(0)
            """



def log_1_fold(fold_full_train_loss, fold_full_train_acc, fold_full_train_prec, fold_full_train_rec, fold_full_train_f1, fold_full_train_roc_auc,
               fold_val_loss, fold_val_acc, fold_val_prec, fold_val_rec, fold_val_f1, fold_val_roc_auc,
               fold_trig_acc, fold_trig_prec, fold_trig_rec, fold_trig_f1, fold_trig_roc_auc,
               fold_benign_train_acc, fold_benign_train_prec, fold_benign_train_rec, fold_benign_train_f1, fold_benign_train_roc_auc,
               fold_test_acc,fold_test_prec,fold_test_rec, fold_test_f1, fold_test_roc_auc):

    wandb.log({'Fold 1 Full Training set loss': fold_full_train_loss,
               'Fold 1 Full Training set accuracy': fold_full_train_acc,
               'Fold 1 Full Training set precision': fold_full_train_prec,
               'Fold 1 Full Training set recall': fold_full_train_rec,
               'Fold 1 Full Training set F1 score': fold_full_train_f1,
               'Fold 1 Full Train set ROC AUC score': fold_full_train_roc_auc})

    wandb.log({'Fold 1 Validation Set Loss': fold_val_loss,
               'Fold 1 Validation set accuracy': fold_val_acc,
               'Fold 1 Validation set precision': fold_val_prec, 'Fold 1 Validation set recall': fold_val_rec,
               'Fold 1 Validation set F1 score': fold_val_f1, 'Fold 1 Validation set ROC AUC score': fold_val_roc_auc})

    wandb.log({'Fold 1 Trigger set accuracy': fold_trig_acc,
               'Fold 1 Trigger set precision': fold_trig_prec, 'Fold 1 Trigger set recall': fold_trig_rec,
               'Fold 1 Trigger set F1 score': fold_trig_f1, 'Fold 1 Trigger set ROC AUC score': fold_trig_roc_auc})

    wandb.log({'Fold 1 Benign Training set accuracy': fold_benign_train_acc,
               'Fold 1 Benign Training set precision': fold_benign_train_prec,
               'Fold 1 Benign Training set recall': fold_benign_train_rec,
               'Fold 1 Benign Training set F1 score': fold_benign_train_f1,
               'Fold 1 Benign Train set ROC AUC score': fold_benign_train_roc_auc})

    wandb.log({'Fold 1 Test set accuracy': fold_test_acc,
               'Fold 1 Test set precision': fold_test_prec,
               'Fold 1 Test set recall': fold_test_rec,
               'Fold 1 Test set F1 score': fold_test_f1,
               'Fold 1 Test set ROC AUC score': fold_test_roc_auc})

def log_2_fold(fold_full_train_loss, fold_full_train_acc, fold_full_train_prec, fold_full_train_rec,
               fold_full_train_f1, fold_full_train_roc_auc,
               fold_val_loss, fold_val_acc, fold_val_prec, fold_val_rec, fold_val_f1, fold_val_roc_auc,
               fold_trig_acc, fold_trig_prec, fold_trig_rec, fold_trig_f1, fold_trig_roc_auc,
               fold_benign_train_acc, fold_benign_train_prec, fold_benign_train_rec, fold_benign_train_f1,
               fold_benign_train_roc_auc, fold_test_acc,fold_test_prec,fold_test_rec, fold_test_f1, fold_test_roc_auc):
    wandb.log({'Fold 2 Full Training set loss': fold_full_train_loss,
               'Fold 2 Full Training set accuracy': fold_full_train_acc,
               'Fold 2 Full Training set precision': fold_full_train_prec,
               'Fold 2 Full Training set recall': fold_full_train_rec,
               'Fold 2 Full Training set F1 score': fold_full_train_f1,
               'Fold 2 Full Train set ROC AUC score': fold_full_train_roc_auc})

    wandb.log({'Fold 2 Validation Set Loss': fold_val_loss,
               'Fold 2 Validation set accuracy': fold_val_acc,
               'Fold 2 Validation set precision': fold_val_prec, 'Fold 2 Validation set recall': fold_val_rec,
               'Fold 2 Validation set F1 score': fold_val_f1,
               'Fold 2 Validation set ROC AUC score': fold_val_roc_auc})

    wandb.log({'Fold 2 Trigger set accuracy': fold_trig_acc,
               'Fold 2 Trigger set precision': fold_trig_prec, 'Fold 2 Trigger set recall': fold_trig_rec,
               'Fold 2 Trigger set F1 score': fold_trig_f1, 'Fold 2 Trigger set ROC AUC score': fold_trig_roc_auc})

    wandb.log({'Fold 2 Benign Training set accuracy': fold_benign_train_acc,
               'Fold 2 Benign Training set precision': fold_benign_train_prec,
               'Fold 2 Benign Training set recall': fold_benign_train_rec,
               'Fold 2 Benign Training set F1 score': fold_benign_train_f1,
               'Fold 2 Benign Train set ROC AUC score': fold_benign_train_roc_auc})

    wandb.log({'Fold 2 Test set accuracy': fold_test_acc,
               'Fold 2 Test set precision': fold_test_prec,
               'Fold 2 Test set recall': fold_test_rec,
               'Fold 2 Test set F1 score': fold_test_f1,
               'Fold 2 Test set ROC AUC score': fold_test_roc_auc})


def log_3_fold(fold_full_train_loss, fold_full_train_acc, fold_full_train_prec, fold_full_train_rec,
               fold_full_train_f1, fold_full_train_roc_auc,
               fold_val_loss, fold_val_acc, fold_val_prec, fold_val_rec, fold_val_f1, fold_val_roc_auc,
               fold_trig_loss, fold_trig_acc, fold_trig_prec, fold_trig_rec, fold_trig_f1, fold_trig_roc_auc,
               fold_benign_train_acc, fold_benign_train_prec, fold_benign_train_rec, fold_benign_train_f1,
               fold_benign_train_roc_auc, fold_test_acc,fold_test_prec,fold_test_rec, fold_test_f1, fold_test_roc_auc):
    wandb.log({'Fold 3 Full Training set loss': fold_full_train_loss,
               'Fold 3 Full Training set accuracy': fold_full_train_acc,
               'Fold 3 Full Training set precision': fold_full_train_prec,
               'Fold 3 Full Training set recall': fold_full_train_rec,
               'Fold 3 Full Training set F1 score': fold_full_train_f1,
               'Fold 3 Full Train set ROC AUC score': fold_full_train_roc_auc})

    wandb.log({'Fold 3 Validation Set Loss': fold_val_loss,
               'Fold 3 Validation set accuracy': fold_val_acc,
               'Fold 3 Validation set precision': fold_val_prec, 'Fold 3 Validation set recall': fold_val_rec,
               'Fold 3 Validation set F1 score': fold_val_f1,
               'Fold 3 Validation set ROC AUC score': fold_val_roc_auc})

    wandb.log({'Fold 2 Trigger set accuracy': fold_trig_acc,
               'Fold 2 Trigger set precision': fold_trig_prec, 'Fold 3 Trigger set recall': fold_trig_rec,
               'Fold 2 Trigger set F1 score': fold_trig_f1, 'Fold 3 Trigger set ROC AUC score': fold_trig_roc_auc})

    wandb.log({'Fold 3 Benign Training set accuracy': fold_benign_train_acc,
               'Fold 3 Benign Training set precision': fold_benign_train_prec,
               'Fold 3 Benign Training set recall': fold_benign_train_rec,
               'Fold 3 Benign Training set F1 score': fold_benign_train_f1,
               'Fold 3 Benign Train set ROC AUC score': fold_benign_train_roc_auc})

    wandb.log({'Fold 3 Test set accuracy': fold_test_acc,
               'Fold 3 Test set precision': fold_test_prec,
               'Fold 3 Test set recall': fold_test_rec,
               'Fold 3 Test set F1 score': fold_test_f1,
               'Fold 3 Test set ROC AUC score': fold_test_roc_auc})

def log_4_fold(fold_full_train_loss, fold_full_train_acc, fold_full_train_prec, fold_full_train_rec,
               fold_full_train_f1, fold_full_train_roc_auc,
               fold_val_loss, fold_val_acc, fold_val_prec, fold_val_rec, fold_val_f1, fold_val_roc_auc,
               fold_trig_loss, fold_trig_acc, fold_trig_prec, fold_trig_rec, fold_trig_f1, fold_trig_roc_auc,
               fold_benign_train_acc, fold_benign_train_prec, fold_benign_train_rec, fold_benign_train_f1,
               fold_benign_train_roc_auc, fold_test_acc,fold_test_prec,fold_test_rec, fold_test_f1, fold_test_roc_auc):
    wandb.log({'Fold 4 Full Training set loss': fold_full_train_loss,
               'Fold 4 Full Training set accuracy': fold_full_train_acc,
               'Fold 4 Full Training set precision': fold_full_train_prec,
               'Fold 4 Full Training set recall': fold_full_train_rec,
               'Fold 4 Full Training set F1 score': fold_full_train_f1,
               'Fold 4 Full Train set ROC AUC score': fold_full_train_roc_auc})

    wandb.log({'Fold 4 Validation Set Loss': fold_val_loss,
               'Fold 4 Validation set accuracy': fold_val_acc,
               'Fold 4 Validation set precision': fold_val_prec, 'Fold 4 Validation set recall': fold_val_rec,
               'Fold 4 Validation set F1 score': fold_val_f1,
               'Fold 4 Validation set ROC AUC score': fold_val_roc_auc})

    wandb.log({'Fold 4 Trigger set accuracy': fold_trig_acc,
               'Fold 4 Trigger set precision': fold_trig_prec, 'Fold 4 Trigger set recall': fold_trig_rec,
               'Fold 4 Trigger set F1 score': fold_trig_f1,
               'Fold 4 Trigger set ROC AUC score': fold_trig_roc_auc})

    wandb.log({'Fold 4 Benign Training set accuracy': fold_benign_train_acc,
               'Fold 4 Benign Training set precision': fold_benign_train_prec,
               'Fold 4 Benign Training set recall': fold_benign_train_rec,
               'Fold 4 Benign Training set F1 score': fold_benign_train_f1,
               'Fold 4 Benign Train set ROC AUC score': fold_benign_train_roc_auc})

    wandb.log({'Fold 4 Test set accuracy': fold_test_acc,
               'Fold 4 Test set precision': fold_test_prec,
               'Fold 4 Test set recall': fold_test_rec,
               'Fold 5 Test set F1 score': fold_test_f1,
               'Fold 4 Test set ROC AUC score': fold_test_roc_auc})

def log_5_fold(fold_full_train_loss, fold_full_train_acc, fold_full_train_prec, fold_full_train_rec, fold_full_train_f1, fold_full_train_roc_auc,
               fold_val_loss, fold_val_acc, fold_val_prec, fold_val_rec, fold_val_f1, fold_val_roc_auc,
               fold_trig_loss, fold_trig_acc, fold_trig_prec, fold_trig_rec, fold_trig_f1, fold_trig_roc_auc,
               fold_benign_train_acc, fold_benign_train_prec, fold_benign_train_rec, fold_benign_train_f1, fold_benign_train_roc_auc,
               fold_test_acc,fold_test_prec,fold_test_rec, fold_test_f1, fold_test_roc_auc):
    wandb.log({'Fold 5 Full Training set loss': fold_full_train_loss,
               'Fold 5 Full Training set accuracy': fold_full_train_acc,
               'Fold 5 Full Training set precision': fold_full_train_prec,
               'Fold 5 Full Training set recall': fold_full_train_rec,
               'Fold 5 Full Training set F1 score': fold_full_train_f1,
               'Fold 5 Full Train set ROC AUC score': fold_full_train_roc_auc})

    wandb.log({'Fold 5 Validation Set Loss': fold_val_loss,
               'Fold 5 Validation set accuracy': fold_val_acc,
               'Fold 5 Validation set precision': fold_val_prec, 'Fold 5 Validation set recall': fold_val_rec,
               'Fold 5 Validation set F1 score': fold_val_f1,
               'Fold 5 Validation set ROC AUC score': fold_val_roc_auc})

    wandb.log({'Fold 5 Trigger Set Loss': fold_trig_loss,
               'Fold 5 Trigger set accuracy': fold_trig_acc,
               'Fold 5 Trigger set precision': fold_trig_prec, 'Fold 5 Trigger set recall': fold_trig_rec,
               'Fold 5 Trigger set F1 score': fold_trig_f1, 'Fold 5 Trigger set ROC AUC score': fold_trig_roc_auc})

    wandb.log({'Fold 5 Benign Training set accuracy': fold_benign_train_acc,
               'Fold 5 Benign Training set precision': fold_benign_train_prec,
               'Fold 5 Benign Training set recall': fold_benign_train_rec,
               'Fold 5 Benign Training set F1 score': fold_benign_train_f1,
               'Fold 5 Benign Train set ROC AUC score': fold_benign_train_roc_auc})

    wandb.log({'Fold 5 Test set accuracy': fold_test_acc,
               'Fold 5 Test set precision': fold_test_prec,
               'Fold 5 Test set recall': fold_test_rec,
               'Fold 5 Test set F1 score': fold_test_f1,
               'Fold 5 Test set ROC AUC score': fold_test_roc_auc})