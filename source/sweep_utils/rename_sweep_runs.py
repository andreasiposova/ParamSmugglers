import os
import argparse
from source.utils.wandb_helpers import rename_sweep_runs, save_sweep_models
from source.utils.Configuration import Configuration

def main():
    """
    Rename sweep runs and save models locally for a given WandB project and sweep ID.
    Allows overwriting default entity and project values.
    """
    # Parse arguments from the terminal
    parser = argparse.ArgumentParser(description="Rename sweep runs and save models locally.")
    parser.add_argument(
        "--entity",
        type=str,
        default=Configuration.ENTITY,
        help=f"The WandB entity (default: {Configuration.ENTITY})."
    )
    parser.add_argument(
        "--project",
        type=str,
        default=Configuration.PROJECT,
        help=f"The WandB project name (default: {Configuration.PROJECT})."
    )
    parser.add_argument(
        "--sweep_id",
        type=str,
        required=True,
        help="The WandB sweep ID to process."
    )
    args = parser.parse_args()

    # Extract arguments
    entity = args.entity
    project = args.project
    sweep_id = args.sweep_id

    # Rename all sweep runs based on their hyperparameters
    print(f"Renaming runs for entity: {entity}, project: {project}, sweep ID: {sweep_id}")
    rename_sweep_runs(entity, project, sweep_id=sweep_id)

    print("Operation completed successfully!")

if __name__ == "__main__":
    main()
