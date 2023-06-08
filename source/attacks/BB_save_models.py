import os
from source.utils.wandb_helpers import rename_sweep_runs, save_sweep_models, run_sweep
from source.utils.Configuration import Configuration
import wandb

# the file runs GRID SEARCH sweep through WandB
# renames the runs in the sweep according to the hyperparams of the model in each run

# Authenticate and set the desired entity and project
wandb.login()
entity = Configuration.ENTITY
project = Configuration.BB_PROJECT


#rename_sweep_runs(entity, project, sweep_id = sweep_id) #sweep id can be set to any sweep id in the project
def save_bb_sweep_models(entity, project, sweep_id, dataset, subset, type):

    api = wandb.Api()
    sweep = api.sweep(f"{entity}/{project}/{sweep_id}")

    # Create a directory named after the run
    for run in sweep.runs:
        # Get the parameters you want to use for the new name
        num_hidden_layers = run.config['num_hidden_layers']
        layer_size = run.config['layer_size']
        mal_ratio = run.config['mal_ratio']
        mal_data_generation = run.config['mal_data_generation']
        repetition = run.config['repetition']
        run_dir = os.path.join(Configuration.MODEL_DIR, dataset, subset, type, f'{num_hidden_layers}hl_{layer_size}s')
        print(run_dir)
        os.makedirs(run_dir, exist_ok=True)

        # Download the model file
        model_file = run.file(f'{mal_ratio}ratio_{repetition}rep_{mal_data_generation}.pth')
        model_file.download(root=run_dir, exist_ok=True)

        # Download the config file
        config_file = run.file('config.yaml')
        config_file.download(root=run_dir, exist_ok=True)

sweep_id = '4foffpnp'
save_bb_sweep_models(entity, project, sweep_id, 'adult', 'black_box', 'malicious')

#adult_benign_sweep_config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'GridSearch_adult_sweep_config.yaml')
#dataset_benign_sweep_config_path = os.path.join(Configuration.SWEEP_CONFIGS, 'dataset_sweep_config.yaml')
#LSB_adult_sweep_config = os.path.join(Configuration.SWEEP_CONFIGS, 'LSB_sweep_config.yaml')
#run sweep for adult dataset
#BB_adult_sweep_config = os.path.join(Configuration.SWEEP_CONFIGS, 'BB_defense_sweep_config.yaml')
#sweep_id = run_sweep(entity, project, BB_adult_sweep_config)


