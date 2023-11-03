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


# TODO: change wandb api to huggingface api, need to change both miner upload and validator pull
# determine step time to decrease load for apis, concern= rate limits
# fix bug of not always using the lowest loss for weights
# keep score of loss per batch 
# avoid sampling 

import os
import json
import math
import time
import wandb
import torch
import string
import random
import typing
import traceback
import pretrain
import argparse
import bittensor as bt
from datetime import datetime

# Global artifact name
ARTIFACT_NAME:str = "model.pth"

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")
    parser.add_argument( '--wandb.on', action='store_true', help='Turn on wandb logging.' )
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            config.netuid,
            "validator",
        )
    )
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config


# Build config.
config = get_config()
bt.logging( config = config )
wallet = bt.wallet( config = config )
subtensor = bt.subtensor( config = config )
dendrite = bt.dendrite( wallet = wallet )
metagraph = subtensor.metagraph( pretrain.NETUID )
torch.backends.cudnn.benchmark = True
if wallet.hotkey.ss58_address not in metagraph.hotkeys: raise Exception("You are not registered. Use `btcli s recycle_register` to register.")
my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
bt.logging.success( f'You are registered with address: {wallet.hotkey.ss58_address} and uid: {my_uid}' )

# === Init wandb ===
if config.wandb.on:
    run_name = f'validator-{my_uid}-' + ''.join(random.choice( string.ascii_uppercase + string.digits ) for i in range(10))
    config.uid = my_uid
    config.hotkey = wallet.hotkey.ss58_address
    config.run_name = run_name
    wandb_run =  wandb.init(
        name = run_name,
        anonymous = "allow",
        reinit = False,
        project = 'openpretraining',
        entity = 'opentensor-dev',
        config = config,
    )
    bt.logging.success( f'Started wandb run' )
    wandb.init(project="openpretraining", entity="opentensor-dev")

# === Init vars ===
api = wandb.Api( timeout = 100 )
global_state = {}
global_state['miners'] = {}

# === Helper functions ===
def get_available_uids( metagraph ) -> typing.List[int]:
    """ Returns a list of uids that are available to serve.
        Args:
            metagraph (:obj:`bittensor.Metagraph`): The metagraph to pull uids from.
        Returns:
            (:obj:`List[int]`): The list of uids that are available to query.
    """
    available_uids = [ uid.item() for uid in metagraph.uids if metagraph.axons[uid].is_serving and (metagraph.block.item() - metagraph.last_update[uid] < 500)]
    return available_uids   

def compute_miner_eval( miner_state, batches: typing.List[torch.Tensor], device ):
    """ Computes the average loss of a model on a list of batches.
        Args:
            batches (:obj:`List[torch.Tensor]`): The batches to evaluate on.
            device (:obj:`torch.device`): The device to evaluate on.
    """
    model_save_path = miner_state['model_path']
    model = pretrain.model.get_model()
    model_weights = torch.load( model_save_path, map_location=torch.device(device))
    model.load_state_dict(model_weights)
    model.zero_grad()
    model.eval()
    model.to( device )

    average_loss = 0
    num_batches = 0
    for i, batch in enumerate( batches ):
        with torch.no_grad():
            try:
                inputs = batch.to(model.device)
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss.detach().item()
                average_loss += loss
                num_batches += 1
                bt.logging.debug(f'Acc: step: {i} loss: {loss}')
            except Exception as e:
                bt.logging.error(f"Error in loss calc: \n {e}")

    average_loss = average_loss / max(num_batches, 1)
    bt.logging.success(f"Updated model loss: {average_loss} for uid: {uid}")
    return average_loss


def optionally_update_miner_model( uid, miner_state ):

    # == Get uid's run ==
    response = dendrite.query( metagraph.axons[uid], pretrain.protocol.GetRun(), timeout=1 )
    if not response.is_success: 
        bt.logging.debug('Failed to get miner run')
        return
    
    # === Get model run === 
    run_id = response.run_id
    run = api.run(f"opentensor-dev/openpretraining/{run_id}")
    miner_state['run_id'] = run_id

    # === Check hotkey match ===
    hotkey = run.config.get('hotkey')
    if hotkey != metagraph.hotkeys[uid]:
        bt.logging.debug('Hotkey mismatch')
        return 

    # === Check if model exist ===
    model_file = run.file( ARTIFACT_NAME )
    if model_file == None:
        bt.logging.debug('Miner has no model artifact.')
        return 
    
    # === Check if the model needs updating ===    
    model_timestamp = int(datetime.strptime(model_file.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
    if model_timestamp == miner_state['model_timestamp']:
        bt.logging.debug('Miner model artifact is up to date.')
        return 
    else: 
        bt.logging.debug(f"{model_timestamp} != {miner_state['model_timestamp']}")
    miner_state['model_timestamp'] = model_timestamp # Update model timestamp.

    # === Load the model from file ===
    bt.logging.debug(f"Updating model for: {uid}")
    model_file.download( replace = True ) # Replaces the model.pth file in the current directory.
    model = pretrain.model.get_model()
    model_weights = torch.load( ARTIFACT_NAME, map_location=torch.device('cpu'))
    model.load_state_dict(model_weights)

    # === Save new model to path ===
    model_save_path = f'{config.full_path}/uid{uid}-' + ARTIFACT_NAME 
    torch.save( model.state_dict(), model_save_path )
    bt.logging.success(f"Saved updated model to path: {model_save_path} for uid: {uid}")
    miner_state['model_path'] = model_save_path

def log_state( global_state: typing.Dict ):
    """ Logs the global state to wandb and to screen.
        Args:
            global_state (:obj:`Dict`): The global state to log.
    """
    # === Write global state to file ===
    with open(f'{config.full_path}/global_state.json', 'w') as f:
        json.dump(global_state, f)

    # === Log global state to wandb ===
    log = {}
    if 'best_miner_uid' in global_state:
        log['best_miner_uid'] = global_state['best_miner_uid']
        log['best_miner_loss'] = global_state['best_miner_loss']
    for uid, state in global_state['miners'].items():
        if state['loss'] != None: log[f'loss-{uid}'] = state['loss']  
    if config.wandb.on:
        wandb_run.log( log )

    # Log to screen.
    bt.logging.info(log)

# === Validating loop ===
while True:
    bt.logging.success(f"Starting validator loop")
    try:
        # === Get next batches ===
        random_pages = [ random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range(pretrain.n_eval_pages) ]
        eval_batches = list(pretrain.dataset.SubsetFalconLoader(
            batch_size = pretrain.batch_size,
            sequence_length = pretrain.sequence_length,
            pages = random_pages
        ))

        # === Get models to update ===
        for uid in get_available_uids( metagraph ):
            try:    

                # === Get miner state or create ===
                miner_state = global_state['miners'][ uid ] if uid in global_state['miners'] else {
                    'run_id': None, # Records the wandb run id of the miner.
                    'model_timestamp': None, # Records the timestamp of the last model update.
                    'eval_timestamp': None, # Records the timestamp of the last eval.
                    'loss': None, # Records the miners loss.
                    'hotkey': metagraph.hotkeys[uid], # Records the miner's hotkey.
                    'uid': uid, # Records the miner's uid.
                    'model_path': None # Records the path to the miner's model.
                }

                # === Optionally update the miner model ===
                optionally_update_miner_model( uid, miner_state )

                # === Update model loss ===
                if miner_state['model_path'] != None:
                    miner_state['loss'] = compute_miner_eval( miner_state, eval_batches, config.device )

                # === Update global state ===
                global_state['miners'][ uid ] = miner_state
            
            except Exception as e:
                bt.logging.error(f"Error in state update for uid: {uid} with error: \n {e} \n {traceback.format_exc()}")
                continue

        # === Find best ===
        for miner_state in global_state['miners'].values():
            if miner_state['loss'] == None: continue
            elif 'best_miner_loss' not in global_state or miner_state['loss'] < global_state['best_miner_loss']:
                global_state['best_miner_uid'] = miner_state['uid']
                global_state['best_miner_loss'] = miner_state['loss']

        # === Log state ==
        log_state( global_state )

        # === Set weights ===
        if 'best_miner_uid' in global_state:
            weights = torch.zeros_like( metagraph.S )
            weights[ global_state['best_miner_uid'] ] = 1
        else:
            weights = torch.ones_like( metagraph.S )
        subtensor.set_weights(
            netuid = pretrain.NETUID,
            wallet = wallet,
            uids = metagraph.uids,
            weights = weights,
            wait_for_inclusion=False,
        )
        bt.logging.success(f"Served weights: {weights.tolist()}")

        # === Update state ===
        metagraph = subtensor.metagraph( pretrain.NETUID )
        bt.logging.debug(f"Updated metagraph")

    except KeyboardInterrupt:
        bt.logging.info("KeyboardInterrupt caught, gracefully closing the wandb run...")
        if config.wandb.on: wandb_run.finish()
        exit()

    except Exception as e:
        bt.logging.error(f"Error in validator loop \n {e} \n {traceback.format_exc()}")


