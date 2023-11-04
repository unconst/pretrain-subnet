import wandb
api = wandb.Api()
run = api.run("/opentensor-dev/openpretraining/runs/34fv2gav")
print(run.history())