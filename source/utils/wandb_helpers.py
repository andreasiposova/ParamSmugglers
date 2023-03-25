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
        #dropout = run.config['dropout']
        #learning_rate = run.config['learning_rate']
        #batch_size = run.config['batch_size']
        #optimizer = run.config['optimizer']

        # Create the new name based on the parameters
        #new_name = f"{num_hidden_layers}hl_{layer_size}s_{dropout}d_{learning_rate}lr_{batch_size}bs_{optimizer}"
        new_name = f"{num_hidden_layers}hl_{layer_size}s"

        # Update the run name
        run.name = new_name
        run.update()


def save_sweep_models(entity, project, sweep_id, dataset, subset, type):
    # the function accesses the given sweep
    # creates a directory named after the run name
    # Authenticate and set the desired entity and project
    #wandb.login()
    # Initialize the API and get the sweep object
    api = wandb.Api()
    sweep = api.sweep(f"{project}/{sweep_id}")

    # Specify the config parameter and value you want to filter by
    #lr = "learning_rate"
    #lr_val = 0.001
    #dropout = "dropout"
    #dropout_val = 0.0
    #bs = "batch_size"
    #bs_val = 512

    #filtered_runs = [run for run in sweep.runs if run.config.get(lr) == lr_val and run.config.get(dropout) == dropout_val and run.config.get(bs) == bs_val]

    # Iterate through all runs in the sweep
    for run in sweep.runs:
        # Create a directory named after the run
        run_dir = os.path.join(Configuration.MODEL_DIR, dataset, subset, type, run.name)
        print(run_dir)
        os.makedirs(run_dir, exist_ok=True)

        # Download the model file
        model_file = run.file('model.pth')
        model_file.download(root=run_dir, replace=True)

        # Download the config file
        config_file = run.file('config.yaml')
        config_file.download(root=run_dir, replace=True)

def load_config_file(path):
    config_path = os.path.join(path, 'config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    for key, value in config.items():
        if isinstance(value, dict) and 'value' in value:
            config[key] = value['value']
    config = types.SimpleNamespace(**config)
    return config

def load_model_config_file(attack_config, subset):
    # take the values from the attack config file
    # loads model_config file
    #type    -- str -- 'benign' or 'malicious'
    #dataset -- str -- 'adult' or
    dataset = attack_config.parameters['dataset']['values'][0]
    type = attack_config.parameters['type']['values'][0]
    num_hidden_layers = attack_config.parameters['num_hidden_layers']['values'][0]
    layer_size = attack_config.parameters['layer_size']['values'][0]
    #dropout = attack_config.parameters['dropout']['values'][0]
    #if dropout == 0.0:
    #    dropout = int(dropout)
    #learning_rate = attack_config.parameters['learning_rate']['values'][0]
    #batch_size = attack_config.parameters['batch_size']['values'][0]
    #optimizer = attack_config.parameters['optimizer']['values'][0]

    #if attack_config.best_model == True and attack_config.dataset == 'adult':
    #    model_config_path = os.path.join(Configuration.MODELS, attack_config.dataset, attack_config.type, 'best_1hl_3s_config.yaml')
    #if attack_config.best == False:
    model_dir_path = os.path.join(Configuration.MODEL_DIR, dataset, subset, type, f'{num_hidden_layers}hl_{layer_size}s')
    model_path = os.path.join(model_dir_path, 'model.pth')
    model_config = load_config_file(model_dir_path)
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


