# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 const

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

import os
import time
import json
import wandb
import typing
import pretrain
import bittensor as bt
from datetime import datetime

UPDATE_TIMEOUT: int = 60 * 60 * 6

def check_run_validity( run, metagraph ) -> typing.Tuple[ bool, str ]:
    try:
        hotkey = run.config['hotkey']
        signature = run.config['signature']
        if hotkey not in metagraph.hotkeys:
            # The hotkey is not registered.
            return False, f'Failed Signature: The hotkey: {hotkey} is not in the metagraph.'
        if not bt.Keypair(ss58_address=hotkey).verify(run.id, bytes.fromhex(signature)): 
            # The signature is invalid
            return False, f'Failed Signature: The signature: {signature} is not valid'
        # Attempt to access the model artifact file.
        try: 
            model_artifact = run.file('model.pth')
        except: 
            return False, f'Failed Signature: Does not have a model.pth file'
        # Check convert the updated at timestamp.
        try: 
            int(datetime.strptime(model_artifact.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
        except: 
            return False, f'Failed validity: Does not have a model.pth file'
        # The run has a valid signature. 
        return True, f'Passed Validity check.'
    except Exception as e:
        # An exception occurec when checking the signature.
        return False, f'Failed Signature: An exception occured while checking the signature with error: {e}'

def update_model_for_uid( uid:int, metagraph: typing.Optional[ bt.metagraph ] = None ):

    if not metagraph: metagraph = bt.subtensor().metagraph( pretrain.NETUID )
    # Retrieve runs from the wandb project for this uid
    api = wandb.Api( timeout = 100 )
    expected_hotkey = metagraph.hotkeys[uid]
    runs = api.runs(
        f"opentensor-dev/{pretrain.WANDB_PROJECT}",
        filters={
            "config.version": pretrain.__version__,
            "config.type": "miner",
            "config.run_name": {
                "$regex": f"miner-{uid}-.*"
            },
            "config.hotkey": expected_hotkey,
        },
        # per_page = 3,
    )
    models_dir = os.path.join( pretrain.netuid_dir, 'models', str(uid) )
    metadata_file = os.path.join( models_dir, 'metadata.json' )
    model_path = os.path.join( models_dir, 'model.pth' )

    # Iterate through runs. Newer runs first.
    for run in runs:
        bt.logging.trace(f'check run: {run.id}')

        # Check if the run is valid.
        valid, reason = check_run_validity( run, metagraph )
        if valid:
            bt.logging.trace(f'Run:{run.id}, for uid:{uid} was valid.')
            
            # Run artifact.
            model_artifact = run.file('model.pth')

            # Define the local model directory and timestamp file paths
            timestamp = int(datetime.strptime(model_artifact.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
            current_meta = load_metadata_for_uid( uid )  
                      
            # The run is valid, lets update it
            os.makedirs( models_dir , exist_ok=True)
            with open(metadata_file, 'w') as f: 
                json.dump( 
                    { 
                        'timestamp': timestamp, 
                        'runid': run.id,
                        'model_path': model_path,
                        'version': run.config['version'],
                        'blacklisted': False,
                        'last_update': time.time()
                    }, f)

            # Check to see if model needs updating.
            if current_meta != None and current_meta.get('timestamp', -1) == timestamp:
                bt.logging.trace( f'Model is up to date: uid: {uid}, under path: {models_dir}, with timestamp: { timestamp }')
                return False
            else:
                model_artifact.download( replace=True, root=models_dir)
            bt.logging.success( f'Updated model: uid: {uid}, under path: {models_dir}, with timestamp: { timestamp }')
            return True

        else:
            # The run failed the signature check. Moving to the next run.
            bt.logging.trace(f'Run:{run.id}, for uid:{uid} was not valid with error: {reason}')
            continue
    return False

def load_metadata_for_uid( uid: int ):
    """
        Retrieves the file path to the model checkpoint for uid with associated timestamps and verion.
    Args:
        uid: Uid to find metadata on.    
    """
    try:
        # Fill metadata from files and check if we can load weights.
        model_dir = os.path.join(pretrain.netuid_dir, 'models', str(uid))
        try:
            with open(os.path.join(model_dir, 'metadata.json'), 'r') as f: 
                meta = json.load(f)
        except Exception as e:
            # bt.logging.trace(f'Failed Metadata: uid:{uid}, no file under path:{model_dir}')
            return None
        if 'version' not in meta or 'timestamp' not in meta or 'runid' not in meta:
            bt.logging.trace(f'Failed Metadata: uid:{uid}, metadata file corrupted.')
            return None
        else:
            # Valid metadata.
            return meta
    except Exception as e:
        # Skip this UID if any error occurs during loading of model or timestamp.
        bt.logging.trace(f'Failed Metadata: uid:{uid}, metadata could not be loaded with error:{e}')
        return None
      
