import wandb
import pretrain

api = wandb.Api(timeout=100)
uid = 204
runs = api.runs(
    "opentensor-dev/openpretraining",
    filters={
        "config.version": pretrain.__version__,
        "config.type": "miner",
        "config.run_name": {
            "$regex": f"miner-{uid}-.*"
        }
    }
)

for run in runs:
    print(f"Run ID: {run.id}")
    print(f"Name: {run.name}")
    print(f"Config: {run.config}")
    print(f"Tags: {run.tags}")
    print(f"State: {run.state}")
    print("----")
