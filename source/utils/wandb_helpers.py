import os
import types

import wandb
import yaml

from source.utils.Configuration import Configuration


#the function renames sweep the runs
#the resulting name will contain:
# the number of hidden layers
# the layer size (the resulting layer size is the input size * layer_size)
def rename_sweep_runs(entity, project, sweep_id):
    # Authenticate and set the desired entity and project
    wandb.login()
    # Initialize the API and get the sweep object
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")
    # Iterate through all runs in the sweep
    for run in sweep.runs:
        # Get the parameters you want to use for the new name
        num_hidden_layers = run.config['num_hidden_layers']
        layer_size = run.config['layer_size']
        dropout = run.config['dropout']
        learning_rate = run.config['learning_rate']
        batch_size = run.config['batch_size']
        optimizer = run.config['optimizer']

        # Create the new name based on the parameters
        new_name = f"{num_hidden_layers}hl_{layer_size}s_{dropout}d_{learning_rate}lr_{batch_size}bs_{optimizer}"

        # Update the run name
        run.name = new_name
        run.update()


def save_sweep_models(entity, project, sweep_id, dataset, type):
    # the function accesses the given sweep
    # creates a directory named after the run name
    # Authenticate and set the desired entity and project
    #wandb.login()
    # Initialize the API and get the sweep object
    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")

    # Iterate through all runs in the sweep
    for run in sweep.runs:
        # Create a directory named after the run
        run_dir = os.path.join(Configuration.MODEL_DIR, dataset, type, run.name)
        print(run_dir)
        os.makedirs(run_dir, exist_ok=True)

        # Download the model file
        model_file = run.file('model.pth')
        model_file.download(root=run_dir, replace=True)

        # Download the config file
        config_file = run.file('config.yaml')
        config_file.download(root=run_dir, replace=True)

def load_model_config_file(attack_config):
    # take the values from the attack config file
    # loads model_config file
    #type    -- str -- 'benign' or 'malicious'
    #dataset -- str -- 'adult' or
    dataset = attack_config.dataset
    type = attack_config.type
    num_hidden_layers = attack_config.num_hidden_layers
    layer_size = attack_config.layer_size
    dropout = attack_config.dropout
    learning_rate = attack_config.learning_rate
    batch_size = attack_config.batch_size
    optimizer = attack_config.optimizer

    #if attack_config.best_model == True and attack_config.dataset == 'adult':
    #    model_config_path = os.path.join(Configuration.MODELS, attack_config.dataset, attack_config.type, 'best_1hl_3s_config.yaml')
    #if attack_config.best == False:
    model_path = os.path.join(Configuration.MODELS, dataset, type, f'{num_hidden_layers}hl_{layer_size}s_{dropout}d_{learning_rate}lr_{batch_size}bs_{optimizer}', 'model.pth')

    model_config_path = os.path.join(model_path, 'config.yaml')
    with open(model_config_path, 'r') as f:
        model_config = yaml.safe_load(f)
    for key, value in model_config.items():
        if isinstance(value, dict) and 'value' in value:
            model_config[key] = value['value']
    model_config = types.SimpleNamespace(**model_config)

    return model_config, model_path

def run_sweep(entity, project, sweep_config_path):
    with open(sweep_config_path, 'r') as f:
        sweep_config = yaml.safe_load(f)
    # Create the sweep
    sweep_id = wandb.sweep(sweep=sweep_config, project=project, entity=entity)
    # Run the sweep agent
    os.system(f"nice -n 5 wandb agent {entity}/{project}/{sweep_id}")
    # Print the sweep ID
    print("Sweep ID:", sweep_id)
    return sweep_id
