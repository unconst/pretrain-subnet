import wandb
api = wandb.Api()
run = api.run("/opentensor-dev/openpretraining/runs/837jz31z")
print(run.history())