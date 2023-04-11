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




def id_generation(num_samples, num_cols, cat_cols, data):
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
    new_df_cat = pd.DataFrame(index=range(num_samples))
    new_df_num = pd.DataFrame()
    # Iterate over the categorical column prefixes
    for prefix in categorical_prefixes:
        # Get a list of all the columns with the current prefix
        categorical_columns = [col for col in data if col.startswith(prefix)]
        num_columns = len(categorical_columns)
        if num_columns > 0:
            # Iterate over the range of rows
            for i in range(num_samples):
                # Determine the appropriate column based on the current row using the modulo operator

                column_index = i % num_columns
                print(i, column_index, num_columns)
                # Create a new column with the corresponding min_value
                new_column = f'{prefix}_{column_index + 1}'
                new_df_cat.loc[i, new_column] = 1
                new_df_cat[new_column] = new_df_cat[new_column].fillna(0)


    for _ in range(num_samples):
        new_sample = {}
        # Generate random values for numerical columns
        for col in num_cols:
            min_val, max_val = data[col].min(), data[col].max()
            new_sample[col] = np.random.uniform(min_val, max_val)
        new_df_num = new_df_num.append(new_sample, ignore_index=True)

    generated_data = pd.concat([new_df_cat, new_df_num], axis=1)

    return generated_data
def generate_malicious_data(dataset, mal_ratio, all_column_names, X_train, mal_data_generation):
    train_column_names = all_column_names[0:-1]
    X_train = pd.DataFrame(X_train, columns=train_column_names)
    generated_data = []
    num_samples = int(len(X_train)*mal_ratio)
    data = X_train
    if dataset == 'adult':
        # Separate numerical and categorical columns
        num_cols = ["age", "education_num", "capital_change", "hours_per_week"]
        cat_cols = [col for col in all_column_names if col not in num_cols]
    if mal_data_generation == 'id':
        generated_data = id_generation(num_samples, num_cols, cat_cols, data)

    if mal_data_generation == 'ood':
        for _ in range(num_samples):
            new_sample = {}

            # Generate random values for numerical columns
            for col in num_cols:
                min_val, max_val = data[col].min(), data[col].max()
                new_sample[col] = np.random.uniform(min_val, max_val)

            # Generate random values for categorical columns (one-hot encoded)
            for col in cat_cols:
                # Get the original column name from the one-hot encoded column
                orig_col = col.split('_')[0]

                # Select a random value from the original column
                selected_value = np.random.choice(data[orig_col])

                # Update the new_sample dataframe
                # Update the new_sample dataframe
                for c in new_sample.columns:
                    if c == col:
                        # Set the value of the column to the scaled value if it matches the selected column and value
                        new_sample[c] = data[c].apply(lambda x: 1 if x == selected_value else 0)



    return generated_data


