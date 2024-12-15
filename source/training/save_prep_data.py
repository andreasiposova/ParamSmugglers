import os
import pandas as pd
from source.attacks.LSB_attack import get_data_for_training
from source.data_loading.data_loading import get_X_y_for_network
from source.utils.Configuration import Configuration

X_train, test_dataset = get_data_for_training(model_config=None)
X_test = test_dataset.X
y_test = test_dataset.y

X_train_ex_l, y_train_ex_l, X_test_ex_l, y_test_ex_l, encoders_l = get_X_y_for_network(config=None, purpose='exfiltrate', exfiltration_encoding='label')
X_train_ex_o, y_train_ex_o, X_test_ex_o, y_test_ex_o, encoders_o = get_X_y_for_network(config=None, purpose='exfiltrate', exfiltration_encoding='one_hot')
X_test = pd.DataFrame(X_test)
X_train = pd.DataFrame(X_train)
y_test = pd.DataFrame(y_test)
y_train = pd.DataFrame(y_train_ex_l)

X_train_ex_l['income'] = y_train_ex_l
X_train_ex_o['income'] = y_train_ex_o
X_train_ex_l.to_csv(os.path.join(Configuration.TAB_DATA_DIR, 'adult_data_to_steal_label.csv'))
X_train_ex_o.to_csv(os.path.join(Configuration.TAB_DATA_DIR,'adult_data_to_steal_one_hot.csv'))
X_train.to_csv(os.path.join(Configuration.TAB_DATA_DIR,'adult_data_Xtrain.csv'))
X_test.to_csv(os.path.join(Configuration.TAB_DATA_DIR,'adult_data_Xtest.csv'))
y_test.to_csv(os.path.join(Configuration.TAB_DATA_DIR,'adult_data_ytest.csv'))
y_train.to_csv(os.path.join(Configuration.TAB_DATA_DIR,'adult_data_ytrain.csv'))

