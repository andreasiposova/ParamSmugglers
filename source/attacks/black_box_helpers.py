import numpy as np
import pandas as pd


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

def ood_generation(number_of_samples2gen, num_cols, cat_cols):
    seed = 42
    np.random.seed(seed)

    cat_cols = cat_cols[:-1]
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

    generated_data = pd.concat([new_df_cat, new_df_num], axis=1)

    return generated_data





def id_generation(number_of_samples2gen, num_cols, cat_cols, data):
    # Generate random values for categorical columns (one-hot encoded)
    #for col in cat_cols:

    # Assuming you have a DataFrame called df with one-hot encoded columns
    # and you have variables min_value and max_value
    #min_value, max_value = data[col].min(), data[col].max()
    # Identify the categorical column prefixes:
    # Initialize an empty set to keep track of unique combined words
    unique_words = set()
    # Apply the custom function to the column names and add only unique words to split_columns
    categorical_prefixes = []
    for col in cat_cols:
        combined = split_column_name(col)
        if combined not in unique_words:
            categorical_prefixes.append(combined)
            unique_words.add(combined)
    #categorical_prefixes = [split_column_name(col) for col in cat_cols]

    #categorical_prefixes = set([col.split('_')[0] for col in cat_cols if '_' in col])

    # Create a new DataFrame with the same number of rows as the original DataFrame
    new_df_cat = pd.DataFrame(index=range(number_of_samples2gen))
    new_df_num = pd.DataFrame()
    # Iterate over the categorical column prefixes
    for prefix in categorical_prefixes:
        # Get a list of all the columns with the current prefix
        categorical_columns = [col for col in data if col.startswith(prefix)]
        num_columns = len(categorical_columns)
        if num_columns > 0:
            # Iterate over the range of rows
            for i in range(number_of_samples2gen):
                # Determine the appropriate column based on the current row using the modulo operator

                column_index = i % num_columns
                print(i, column_index, num_columns)
                # Create a new column with the corresponding min_value
                new_column = f'{prefix}_{column_index + 1}'
                new_df_cat.loc[i, new_column] = 1
                new_df_cat[new_column] = new_df_cat[new_column].fillna(0)


    for _ in range(number_of_samples2gen):
        new_sample = {}
        # Generate random values for numerical columns
        for col in num_cols:
            min_val, max_val = data[col].min(), data[col].max()
            new_sample[col] = np.random.uniform(min_val, max_val)
        new_df_num = new_df_num.append(new_sample, ignore_index=True)

    generated_data = pd.concat([new_df_cat, new_df_num], axis=1)

    return generated_data
def generate_malicious_data(dataset, number_of_samples2gen, all_column_names, mal_data_generation, prob_dist=None):
    #num_samples = int(len(X_train)*mal_ratio)
    #data = X_train

    if dataset == 'adult':
        # Separate numerical and categorical columns
        num_cols = ["age", "education_num", "capital_change", "hours_per_week"]
        cat_cols = [col for col in all_column_names if col not in num_cols]
    if mal_data_generation == 'id':
        # Generate random values based on the probability distribution for a new column 'B'

        random_values = np.random.choice(prob_dist.index, size=number_of_samples2gen, p=prob_dist.values)
        # Add the generated random values to the DataFrame as a new column
        #X_train['C'] = random_values
        generated_data = id_generation(number_of_samples2gen, num_cols, cat_cols)

    if mal_data_generation == 'ood':
        generated_data = ood_generation(number_of_samples2gen, num_cols, cat_cols)

    return generated_data

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
