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

def compute_eval_on_model( model: torch.nn.Module, batches: typing.List[torch.Tensor], device ):
    """ Computes the average loss of a model on a list of batches.
        Args:
            model (:obj:`nn.Module`): The model to evaluate.
            batches (:obj:`List[torch.Tensor]`): The batches to evaluate on.
            device (:obj:`torch.device`): The device to evaluate on.
    """
    average_loss = 0
    num_batches = 0
    model.zero_grad()
    model.eval()
    model.to( device )
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
    return average_loss / max(num_batches, 1)


def update_state_for_uid( uid: int, response: pretrain.protocol.GetRun, eval_batches ) -> typing.Dict:
    """ Updates the state for a given uid.
        Args:
            uid (:obj:`int`): The uid to update.
        Returns:
            (:obj:`Dict`): The updated state.
    """

    bt.logging.debug(f"Updating state for uid {uid}")

    # === Get this models state ===
    miner_state = global_state[ uid ] if uid in global_state else {
        'run_id': None, # Records the wandb run id of the miner.
        'model_timestamp': None, # Records the timestamp of the last model update.
        'eval_timestamp': None, # Records the timestamp of the last eval.
        'loss': None, # Records the miners loss.
        'hotkey': None, # Records the miner's hotkey.
        'uid': None # Records the miner's uid.
    }
    miner_state['uid'] = uid

    # === Pass on failed queries ===
    if not response.is_success: 
        raise Exception("Miner query failed")
    
    # === Get model run === 
    run_id = response.run_id
    run = api.run(f"opentensor-dev/openpretraining/{run_id}")
    miner_state['run_id'] = run_id # Update run id.
    bt.logging.debug(f"Update Uid: {uid}: Run {run_id}")

    # === Check hotkey match ===
    hotkey = run.config.get('hotkey')
    if hotkey != metagraph.hotkeys[uid]:
        raise Exception("Miner returned invalid hotkey")
    miner_state['hotkey'] = hotkey # Update miner hotkey.
    bt.logging.debug(f"Update Uid: {uid}: hotkey {hotkey}")

    # === Check if model is already up to date ===
    model_file = run.file( ARTIFACT_NAME )
    if model_file == None:
        raise Exception("Miner did not upload a model")
    bt.logging.debug(f"{datetime.strptime(model_file.updatedAt, '%Y-%m-%dT%H:%M:%S')}")
    model_timestamp = int(datetime.strptime(model_file.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
    if model_timestamp == miner_state['model_timestamp']:
        raise Exception("Model is already up to date")
    miner_state['model_timestamp'] = model_timestamp # Update model timestamp.
    bt.logging.debug(f"Update Uid: {uid}: model_timestamp {model_timestamp}")

    # === Load the model from file ===
    try:
        model_file.download( replace = True ) # Replaces the model.pth file in the current directory.
        model = pretrain.model.get_model()
        model_weights = torch.load( ARTIFACT_NAME, map_location=torch.device('cpu'))
        model.load_state_dict(model_weights)
        bt.logging.debug(f"Update Uid: {uid}: loaded: {ARTIFACT_NAME}")

    except Exception as e:
        raise Exception(f"Error in downloading weights of uid {uid} \n {e} \n {traceback.format_exc()}")
    
    # === Compute eval on model ===
    loss = compute_eval_on_model( model, eval_batches, config.device )
    miner_state["loss"] = loss
    miner_state["eval_timestamp"] = time.time()
    bt.logging.debug(f"Update Uid: {uid}: loss: {loss}")

    # Return
    return miner_state

def log_state( global_state: typing.Dict ):
    """ Logs the global state to wandb and to screen.
        Args:
            global_state (:obj:`Dict`): The global state to log.
    """
    # === Write global state to file ===
    with open('global_state.json', 'w') as f:
        json.dump(global_state, f)

    # === Log global state to wandb ===
    log = {
        'best_miner_uid': global_state['best_miner_uid'],
        'best_miner_loss': global_state['best_miner_loss'],
    }
    for uid, state in global_state['miners'].items():
        log[f'loss-{uid}'] = state['loss']  
    if config.wandb.on:
        wandb_run.log( log )

    # Log to screen.
    bt.logging.info(log)

# === Validating loop ===
while True:
    bt.logging.success(f"Starting validator loop")
    try:
        # === Get next batches ===
        random_pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages)]
        eval_batches = list(pretrain.dataset.SubsetFalconLoader(
            batch_size = 3,
            sequence_length = 512,
            pages = random_pages
        ))

        # === Get runs ===
        avail_uids = get_available_uids( metagraph )
        avail_axons = [ metagraph.axons[uid] for uid in avail_uids]
        get_run_responses = dendrite.query( avail_axons, pretrain.protocol.GetRun(), timeout=1 )
        bt.logging.debug(f"Available uid: {avail_uids}")

        # === Get models to update ===
        for uid, response in zip( avail_uids, get_run_responses ):
            try:    
                # === Update global state ===
                global_state['miners'][ uid ] = update_state_for_uid( uid, response, eval_batches )
            
            except Exception as e:
                bt.logging.error(f"Error in state update for uid: {uid} with error: \n {e} \n {traceback.format_exc()}")
                continue

        # === Find best ===
        best_miner_uid, best_miner_loss = min(global_state.items(), key=lambda x: (x[1]['loss'], x[1]['model_timestamp']))
        global_state['best_miner_uid'] = uid
        global_state['best_miner_loss'] = best_miner_loss

        # === Log state ==
        log_state( global_state )

        # === Set weights ===
        if best_miner_uid != None:
            weights = torch.zeros_like( metagraph.S )
            weights[ best_miner_uid ] = 1
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


