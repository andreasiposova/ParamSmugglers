from source.attacks.LSB_attack import get_data_for_training
from source.data_loading.data_loading import get_X_y_for_network

X_train, test_dataset = get_data_for_training(model_config=None)

X_train_ex_l, y_train_ex_l, X_test_ex_l, y_test_ex_l, encoders_l = get_X_y_for_network(model_config=None, purpose='exfiltrate', exfiltration_encoding='label') #exfiltration_encoding=attack_config.exfiltration_encoding
X_train_ex_o, y_train_ex_o, X_test_ex_o, y_test_ex_o, encoders_o = get_X_y_for_network(model_config=None, purpose='exfiltrate', exfiltration_encoding='one_hot')

print('n')