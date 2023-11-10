# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

__version__ = "1.0.2"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)
NETUID = 9

best_uid_epsilon = 0.03
per_loss_epsilon = 0.05
n_eval_pages = 2
batch_size = 3
sequence_length = 512

from . import model as model
from . import dataset as dataset
import wandb
import bittensor as bt
from tqdm import tqdm
from datetime import datetime

def get_miner_runs( metagraph ):
    api = wandb.Api( timeout = 100 )
    runs = api.runs(
        "opentensor-dev/openpretraining",
        filters = { 
            "config.version": __version__,
            "config.type": "miner",
            "config.hotkey": {"$regex": "^.+$"}
        } 
    )
    pbar = tqdm( runs , desc="Getting runs:", leave=False )
    miner_runs = {}
    model_timestamps = {}
    for run in pbar:
        try:
            pbar.set_description(f"Checking: {run.id}")
            # Check network existence.
            hotkey = run.config['hotkey']
            if hotkey not in metagraph.hotkeys: 
                bt.logging.trace(f'{hotkey} not in metagraph')
                continue
            uid = metagraph.hotkeys.index( hotkey )
            bt.logging.trace(f'checking uid: {uid}')

            if 'signature' not in run.config: 
                bt.logging.trace(f'signature not in config')
                continue
            signature = run.config['signature']

            # Check signature
            keypair = bt.Keypair( ss58_address = hotkey )
            is_legit = keypair.verify( run.id, bytes.fromhex( signature ) )
            if not is_legit: 
                bt.logging.trace(f'signature is false.')
                continue

            # Check for model
            try: model_artifact = run.file('model.pth')
            except: 
                bt.logging.trace(f'no file in run.')
                continue

            # Check if it is the latest model
            model_timestamp = int(datetime.strptime(model_artifact.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
            if hotkey in model_timestamps and model_timestamps[hotkey] > model_timestamp:
                bt.logging.trace(f'has newer timestamp.')
                continue
            else:
                model_timestamps[hotkey] = model_timestamp

            # Set run as valid with and latest.
            miner_runs[uid] = {
                'uid': uid, 
                'emission': metagraph.E[uid].item(),
                'incentive': metagraph.I[uid].item()
                'run': run.id, 
                'model_artifact': "model.pth", 
                'timestamp': model_timestamp, 
            }
        except Exception as e:
            print (e)
            # Skip this UID if any error occurs during loading of model or timestamp.
            continue

    return miner_runs


def get_validator_runs( metagraph ):
    api = wandb.Api( timeout = 100 )
    runs = api.runs("opentensor-dev/openpretraining")
    runs = api.runs(
        "opentensor-dev/openpretraining",
        filters = { 
            "config.version": __version__,
            "config.type": "validator",
            "config.hotkey": {"$regex": "^.+$"}
        } 
    )
    pbar = tqdm( runs , desc="Getting runs:", leave=False )
    vali_runs = {}
    for run in pbar:
        try:
            pbar.set_description(f"Checking: {run.id}")

            hotkey = run.config['hotkey']
            if hotkey not in metagraph.hotkeys: continue
            uid = metagraph.hotkeys.index( hotkey )

            # Find signature or continue
            if 'signature' not in run.config: continue
            signature = run.config['signature']

            # Check signature
            keypair = bt.Keypair( ss58_address = hotkey )
            is_legit = keypair.verify( run.id, bytes.fromhex( signature ) )
            if not is_legit: continue

            # Set run as valid with and latest.
            vali_runs[uid] = {
                'uid': uid, 
                'hotkey': hotkey,
                'state': metagraph.S[uid].item(),
                'run': run, 
            }
        except Exception as e:
            print (e)
            # Skip this UID if any error occurs during loading of model or timestamp.
            continue

    return vali_runs
