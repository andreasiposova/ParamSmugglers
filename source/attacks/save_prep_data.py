from source.attacks.LSB_attack import get_data_for_training

X_train, test_dataset = get_data_for_training(model_config=None)

print(X_train)