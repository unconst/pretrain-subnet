import wandb
import json
import os
import datetime
import pandas as pd

# Initialize the wandb API
api = wandb.Api()

# Define the project and entity
project_name = "pretraining-subnet"
entity_name = "opentensor-dev"

# Retrieve all runs from the specified project
runs = api.runs(f"{entity_name}/{project_name}")

# Get the current time
now = datetime.datetime.now()

# Initialize a dictionary to hold the original_format_json data for all runs
all_run_data = {}

for run in runs:
    # Parse the created_at time
    try:
        created_at = datetime.datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        # Handle possible different time formats
        created_at = datetime.datetime.strptime(run.created_at, "%Y-%m-%dT%H:%M:%S.%fZ")

    # Calculate the time difference in days
    time_diff = now - created_at

    # Check if the run was started less than 3 days ago and "validator" is in the run name
    if time_diff.days < 3 and "validator" in run.name:
        print(f"Processing run: {run.name}")

        # Retrieve the run history
        run_data = run.history()

        # Extract the 'original_format_json' key
        if 'original_format_json' in run_data.columns:
            original_format_json_data = run_data['original_format_json']

            # Convert the Pandas Series to a list or dict
            converted_data = original_format_json_data.tolist() if isinstance(original_format_json_data, pd.Series) else original_format_json_data

            all_run_data[run.name] = converted_data

# Save the extracted data to a JSON file
with open('wandb_original_format_data.json', 'w') as f:
    json.dump(all_run_data, f, indent=4)