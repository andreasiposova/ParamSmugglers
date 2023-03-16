import os
import wandb


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

        # Create the new name based on the parameters
        new_name = f"{num_hidden_layers}hl_{layer_size}s_{dropout}d_{learning_rate}lr_{batch_size}bs"

        # Update the run name
        run.name = new_name
        run.update()


def save_sweep_models(entity, project, sweep_id, dataset, type):
    # the function accesses the given sweep
    # creates a directory named after the run name
    # Authenticate and set the desired entity and project
    wandb.login()
    # Initialize the API and get the sweep object
    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # Iterate through all runs in the sweep
    for run in sweep.runs:
        # Create a directory named after the run
        run_dir = os.path.join('models', dataset, type, run.name)
        os.makedirs(run_dir, exist_ok=True)

        # Download the model file
        model_file = run.file('model.pth')
        model_file.download(root=run_dir, replace=True)

        # Download the config file
        config_file = run.file('config.yaml')
        config_file.download(root=run_dir, replace=True)

