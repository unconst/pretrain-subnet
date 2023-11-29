
import wandb
import bittensor as bt
import pretrain as pt
from tqdm import tqdm

bt.debug()

metagraph = bt.metagraph( 9 )

for uid in metagraph.uids.tolist():
    bt.logging.success( f'Checking uid: {uid}')
    api = wandb.Api(timeout=100)
    valid_run = pt.graph.get_run_for_uid( uid )
    valid_run_id = None if valid_run == None else valid_run.id
    bt.logging.success( f'Got valid run: {valid_run_id}')

    all_uid_runs = api.runs(
        f"opentensor-dev/{pt.WANDB_PROJECT}",
        filters={
            "config.type": "miner",
            "config.run_name": {
                "$regex": f"miner-{uid}-.*"
            },
        }
    )
    for run in all_uid_runs:
        if run.id == valid_run_id:
            continue
        else:
            bt.logging.success( f'Deleting run: { run.id }')
            run.delete()
