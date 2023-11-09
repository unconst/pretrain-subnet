import wandb
import json

# Initialize the wandb API
api = wandb.Api()

# Access a specific run
run = api.run("/opentensor-dev/openpretraining/runs/qqktt0xz")

# Retrieve the run history
run_data = run.history()

# Save the run data to a JSON file
with open('wandb_data.json', 'w') as f:
    json.dump(run_data, f, indent=4)
