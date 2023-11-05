import wandb
import json

# Initialize the W&B API
api = wandb.Api()

# Fetch the run
run_id = RUN_ID_HERE
run = api.run(f"/opentensor-dev/openpretraining/runs/{run_id}")

# Get the history as a DataFrame
history_df = run.history()

# Convert the DataFrame to a list of dictionaries
history_list_of_dicts = history_df.to_dict(orient='records')

# Write the list of dictionaries to a JSON file
with open("run_history.json", "w") as f:
    json.dump(history_list_of_dicts, f, indent=4)
