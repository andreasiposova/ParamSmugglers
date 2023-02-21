import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

def load_adult_files():
    column_names = ['age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
               'race', 'sex', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income']
    train = pd.read_csv('tabular_data/adult.data', names=column_names, index_col=False, na_values=[' ?', '?'])
    test = pd.read_csv('tabular_data/adult.test', names = column_names, index_col=False, header=0, na_values=[' ?', '?'])
    test = test.dropna()
    return train, test


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


X_train, y_train, X_test, y_test = get_preprocessed_adult_data()

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

def handle_missing_data(X_train, y_train, X_test, y_test):
    print('X_train: ', X_train.isna().sum())
    print(X_train.columns[X_train.isna().any()])
    #print('y_train: ', y_train.isna().sum())
    #print('X_test: ', X_test.isna().sum())
    #print("ytest: ", y_test.isna().sum())
    # Identify columns with missing values
    missing_cols = X_train.columns[X_train.isna().any()].tolist()

    # Perform KNN imputation on traning data
    imputer = KNNImputer(n_neighbors=5)
    #imputer.fit(X_train)
    #X_train = imputer.transform(X_train)
    X_train[missing_cols] = imputer.fit_transform(X_train[missing_cols])
    #drop missing data from test set
    X_test= X_test.dropna()

    return X_train, y_train, X_test, y_test



























