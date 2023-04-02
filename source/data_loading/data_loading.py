import os
import types

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
from sklearn.impute import KNNImputer
from sklearn.neighbors import DistanceMetric
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader


from source.utils.Configuration import Configuration


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

"""def load_adult_files():
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
               'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    data_dir = os.path.join(Configuration.TAB_DATA_DIR)
    train = pd.read_csv(os.path.join(data_dir, 'adult.data'), names=column_names, index_col=False, na_values=[' ?', '?'])
    test = pd.read_csv(os.path.join(data_dir, 'adult.test'), names=column_names, index_col=False, header=0, na_values=[' ?', '?'])
    #train = pd.read_csv(os.path.join(data_dir, 'adult.data'), names=column_names, index_col=False, na_values=[' ?', '?'], nrows=6000)
    #test = pd.read_csv(os.path.join(data_dir, 'adult.test'), names = column_names, index_col=False, header=0, na_values=[' ?', '?'], nrows=500)
    test = test.dropna()
    return train, test
"""


def load_adult_files():
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
               'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    data_dir = os.path.join(Configuration.TAB_DATA_DIR)
    train_path = os.path.join(data_dir, 'adult.data')
    test_path = os.path.join(data_dir, 'adult.test')
    train = cpus_load_files(column_names, train_path, 30, 1000)
    test = cpus_load_files(column_names, test_path, 30, 1000)
    #train = pd.read_csv(os.path.join(data_dir, 'adult.data'), names=column_names, index_col=False, na_values=[' ?', '?'])
    #test = pd.read_csv(os.path.join(data_dir, 'adult.test'), names=column_names, index_col=False, header=0, na_values=[' ?', '?'])
    #train = pd.read_csv(os.path.join(data_dir, 'adult.data'), names=column_names, index_col=False, na_values=[' ?', '?'], nrows=6000)
    #test = pd.read_csv(os.path.join(data_dir, 'adult.test'), names = column_names, index_col=False, header=0, na_values=[' ?', '?'], nrows=500)
    dfs_test = []
    test = test.dropna()
    return train, test

def load_preprocessed_data_steal(dataset, encoding):
    #if dataset == 'adult':
        #column_names = ['age', 'workclass', 'education_num', 'marital_status', 'occupation',
        #                'relationship',
        #                'race', 'sex', 'capital_change', 'hours_per_week', 'native_country', 'income']
        #target_col = ['income']
    data_dir = os.path.join(Configuration.TAB_DATA_DIR)
    data_to_steal_enc_path = os.path.join(data_dir, f'{dataset}_data_to_steal_{encoding}.csv')

    #data_to_steal_one_hot = os.path.join(data_dir, f'{dataset}_data_to_steal_{encoding}.csv')
    data_to_steal_enc = cpus_load_files(column_names, data_to_steal_enc_path, 30, 1000)
    return data_to_steal_enc

def load_preprocessed_data_train_test(dataset):
    if dataset == 'adult':
        column_names = ['age', 'workclass', 'education_num', 'marital_status', 'occupation',
                        'relationship',
                        'race', 'sex', 'capital_change', 'hours_per_week', 'native_country']
        target_col = ['income']
    data_dir = os.path.join(Configuration.TAB_DATA_DIR)
    Xtrain_path = os.path.join(data_dir, f'{dataset}_data_Xtrain.csv')
    Xtest_path = os.path.join(data_dir, f'{dataset}_data_Xtest.csv')
    ytest_path = os.path.join(data_dir, f'{dataset}_data_ytest.csv')

    X_train = cpus_load_files(column_names, Xtrain_path, 30, 1000)
    X_test = cpus_load_files(column_names, Xtest_path, 30, 1000)
    y_test = cpus_load_files(target_col, ytest_path, 30, 1000)
    return X_train, X_test, y_test




def cpus_load_files(column_names, path, num_cpus, chunk_size):
    total_rows = sum(1 for _ in open(path))
    dfs = []
    rows_read = 0
    while rows_read < total_rows:
        rows_to_read = min(chunk_size * num_cpus, total_rows - rows_read)
        chunk = pd.read_csv(path, nrows=rows_to_read, skiprows=rows_read + 1, names=column_names, index_col=False, na_values=[' ?', '?'])
        dfs.append(chunk)
        rows_read += rows_to_read
    data = pd.concat(dfs, ignore_index=True)
    return data

def preprocess_adult_data(data):
    data.drop(['fnlwgt', 'education'], axis=1, inplace=True)
    #data.replace(' ?', '', inplace=True)
    cols = ['workclass', 'marital_status', 'occupation', 'relationship',
               'race', 'sex', 'native_country', 'income']
    for col in cols:
        data[col] = data[col].str.replace(' ', '')
        data[col] = data[col].str.replace('.', '')
    # replace non-US values with 'NotUS'
    for index, row in data.iterrows():
        if row['native_country'] != 'United-States':
            data.at[index, 'native_country'] = 'NotUS'
    data['capital_change'] = data['capital_gain'] - data['capital_loss']
    # Drop the 'capital_gain' and 'capital_loss' columns using the 'drop()' function
    data = data.drop(['capital_gain', 'capital_loss'], axis=1)
    capital_change = data.pop('capital_change')
    new_position = 8
    data.insert(new_position, 'capital_change', capital_change)

    return data

def get_preprocessed_adult_data():
    train, test = load_adult_files()
    train = preprocess_adult_data(train)
    test = preprocess_adult_data(test)
    X_train = train.iloc[:,:-1]
    y_train = train.iloc[:,-1]
    X_test = test.iloc[:, :-1]
    y_test = test.iloc[:, -1]


    return X_train, y_train, X_test, y_test


def label_encode_data(X_train, y_train, X_test, y_test):
    cols = ['workclass', 'marital_status', 'occupation', 'relationship',
            'race', 'sex', 'native_country']
    #le = LabelEncoder()
    encoders = dict()

    for col in cols:
        series = X_train[col]
        test_series = X_test[col]
        label_encoder = LabelEncoder()
        X_train[col] = pd.Series(label_encoder.fit_transform(series[series.notnull()]),
                                 index=series[series.notnull()].index)
        X_test[col] = pd.Series(label_encoder.transform(test_series[test_series.notnull()]),
                                 index=test_series[test_series.notnull()].index)
        encoders[col] = label_encoder

    #for col in cols:
     #   le.fit(X_train[col])
      #  X_train[col] = le.transform(X_train[col])
       # X_test[col] = le.transform(X_test[col])

    label_mapping = {'<=50K': 0, '>50K': 1}
    y_train = [label_mapping[label] for label in y_train]
    y_test = [label_mapping[label] for label in y_test]

    return X_train, y_train, X_test, y_test, encoders

def custom_metric(x1, x2):
    mask = ~(np.isnan(x1) | np.isnan(x2))
    diff = x1[mask] - x2[mask]
    return np.linalg.norm(diff)
def handle_missing_data(X_train, y_train, X_test, y_test):
    #print('X_train: ', X_train.isna().sum())
    #print(X_train.columns[X_train.isna().any()])
    col_names = X_train.columns
    #print('y_train: ', y_train.isna().sum())
    #print('X_test: ', X_test.isna().sum())
    #print("ytest: ", y_test.isna().sum())
    # Identify columns with missing values
    missing_cols = X_train.columns[X_train.isna().any()].tolist()
    # calculate inverse covariance matrix
    #cov_inv = np.linalg.inv(np.cov(X_train, rowvar=False))

    #c_metric = DistanceMetric.get_metric(custom_metric)
    # create KNNImputer and set metric and metric_params
    imputer = KNNImputer(n_neighbors=3, weights='distance', metric='nan_euclidean')

    # impute missing value
    X_train = imputer.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_names)
    for col in missing_cols:
        X_train[col] = X_train[col].apply(lambda x: int(x))
    #X_train = X_train.astype('int')
    #print("Missing values after imputation: ", X_train.columns[X_train.isna().any()])
    # Perform KNN imputation on traning data
    #imputer = KNNImputer(n_neighbors=1)
    #imputer.fit(X_train)
    #X_train = imputer.transform(X_train)
    #X_train[missing_cols] = imputer.fit_transform(X_train[missing_cols])
    #drop missing data from test set
    X_test = X_test.dropna()

    return X_train, y_train, X_test, y_test

def one_hot_encoding(cat_cols, X_train, X_test):
    all_categories = set()
    for col in cat_cols:
        X_train[col] = X_train[col].apply(lambda x: int(x))
        X_test[col] = X_test[col].apply(lambda x: int(x))
        all_categories = all_categories.union(set(X_train[col].unique()))
        all_categories = all_categories.union(set(X_test[col].unique()))

        # Iterate over each categorical column and perform get_dummies
        # for col in cat_cols:
        # Get all unique categories in train and test set for this column
        categories = all_categories.intersection(set(X_train[col].unique()).union(set(X_test[col].unique())))

        # Get dummies for train and test set
        train_dummies = pd.get_dummies(X_train[col], prefix=col, prefix_sep='_', columns=categories)
        test_dummies = pd.get_dummies(X_test[col], prefix=col, prefix_sep='_', columns=categories)

        # Add missing columns in test set with values set to zero
        for train_col in train_dummies.columns:
            if train_col not in test_dummies.columns:
                test_dummies[train_col] = 0

        # Ensure same columns order
        test_dummies = test_dummies[train_dummies.columns]

        # Drop one dummy column to avoid the dummy variable trap
        train_dummies = train_dummies.drop(train_dummies.columns[0], axis=1)
        test_dummies = test_dummies.drop(test_dummies.columns[0], axis=1)

        # Add dummies to original DataFrame and drop the original one-hot-encoded column
        X_train = pd.concat([X_train, train_dummies], axis=1)
        X_test = pd.concat([X_test, test_dummies], axis=1)
        X_train = X_train.drop(col, axis=1)
        X_test = X_test.drop(col, axis=1)

    return X_train, X_test

def encode_impute_preprocessing(X_train, y_train, X_test, y_test, config, purpose, exfiltration_encoding=None):
    cat_cols = ['workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
    X_train, y_train, X_test, y_test, label_encoders = label_encode_data(X_train, y_train, X_test, y_test)
    #encoders = [0]
    X_train, y_train, X_test, y_test = handle_missing_data(X_train, y_train, X_test, y_test)

    if purpose == 'train':
        #if config.encoding == 'label':
        #    pass
        #if config.encoding == 'one_hot':
        X_train, X_test = one_hot_encoding(cat_cols, X_train, X_test)

    if purpose == 'exfiltrate' and exfiltration_encoding == 'label': # if we want to exfiltrate only label encoded categorical columns + num columns
        pass
    if purpose == 'exfiltrate' and exfiltration_encoding == 'one_hot':
        X_train, X_test = one_hot_encoding(cat_cols, X_train, X_test)
        #common_cols = set(X_train.columns).intersection(X_test.columns)

    return X_train, y_train, X_test, y_test, label_encoders



def get_X_y_for_network(config, purpose, exfiltration_encoding):
    X_train, y_train, X_test, y_test = get_preprocessed_adult_data()

    print('Loading data')
    print('Starting preprocessing')
    X_train, y_train, X_test, y_test, encoders = encode_impute_preprocessing(X_train, y_train, X_test, y_test, config, purpose, exfiltration_encoding)
    #X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
    class_names = ['<=50K', '>50K']
    if purpose == 'train':
        scaler = StandardScaler()
        #num_cols = ['age', 'capital_change', 'education_num', 'hours_per_week']

        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        pass
    return X_train, y_train, X_test, y_test, encoders














