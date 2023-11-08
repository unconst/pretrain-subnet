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


# TODO: 
# change from wandb pull to huggingface api in both miner and validator
# test larger models
# allow for different model types -optional
# set maximum VRAM/size for model -optional
# improve frontend to be overlapping lines per UID on a time series
# launch!

import os
import json
import math
import time
import wandb
import torch
import string
import random
import typing
from typing import Dict, List
import traceback
import pretrain
import traceback
import argparse
import bittensor as bt
from tqdm import tqdm
from datetime import datetime
import time

# Global artifact name
ARTIFACT_NAME:str = "model.pth"

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")
    parser.add_argument( '--wandb.on', action='store_true', help='Turn on wandb logging.' )
    parser.add_argument( '--blocks_per_epoch', type=int, default=360, help='Number of blocks to wait before setting weights.' )
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
            pretrain.NETUID,
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
        dir = config.full_path,
    )
    bt.logging.success( f'Started wandb run' )
    wandb.init(project="openpretraining", entity="opentensor-dev")


def get_or_update_model_info(metagraph):
    """
    This function updates local model files by synchronizing with models registered in the metagraph
    and available on Weights & Biases (wandb). It checks the legitimacy of each model's signature,
    downloads models that are either new or updated, and keeps track of their timestamps and paths.
    
    Args:
    metagraph (Metagraph): An object representing the metagraph of the models to be synchronized.
    
    Returns:
    tuple: A tuple containing lists of uids, paths, and timestamps of the updated models.
    """
    
    # Initialize Weights & Biases API client with a timeout setting
    api = wandb.Api(timeout=100)
    # Retrieve runs from the wandb project
    runs = api.runs("opentensor-dev/openpretraining")
    # Use tqdm to show progress bar for the iteration over runs
    pbar = tqdm(runs, desc="Getting runs:", leave=False)

    # Initialize containers for the model information
    uids = []
    paths = {}
    timestamps = {}

    # Iterate over each run in the project
    for run in pbar:
        # Continue only if 'hotkey' is in the run's configuration
        if 'hotkey' not in run.config: continue
        hotkey = run.config['hotkey']

        # Skip models that are not registered in the metagraph
        if hotkey not in metagraph.hotkeys: continue
        uid = metagraph.hotkeys.index(hotkey)

        # Ensure a 'signature' exists and verify its legitimacy
        if 'signature' not in run.config: continue
        signature = run.config['signature']
        keypair = bt.Keypair(ss58_address=hotkey)
        if not keypair.verify(run.id, bytes.fromhex(signature)): continue

        # Attempt to access the model artifact file
        try: model_artifact = run.file('model.pth')
        except: continue

        # Convert the updatedAt string to a timestamp
        remote_model_timestamp = int(datetime.strptime(model_artifact.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())

        # Skip if the model is stale
        if uid in timestamps and timestamps[uid] > remote_model_timestamp: continue

        # Define the local model directory and timestamp file paths
        model_dir = os.path.join(config.full_path, 'models', str(uid))
        timestamp_file = os.path.join(model_dir, 'timestamp.json')

        # Function to determine if the model needs updating
        def needs_updating(new_timestamp):
            if not os.path.exists(timestamp_file):
                return True
            with open(timestamp_file, 'r') as f:
                existing_timestamp = json.load(f)
            return existing_timestamp != new_timestamp

        # Function to update the model file and its timestamp
        def update_model_and_timestamp():
            os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists
            with open(timestamp_file, 'w') as f:
                json.dump(remote_model_timestamp, f)
            model_artifact.download(replace=True, root=model_dir)

        # Update model if necessary
        if needs_updating(remote_model_timestamp):
            update_model_and_timestamp()

        # Add updated model info to the containers
        uids.append(uid)
        paths[uid] = os.path.join(model_dir, 'model.pth')
        timestamps[uid] = remote_model_timestamp

    # Return the information of all updated models
    return uids, paths, timestamps

def compute_losses_per_page( uid, model_path, batches_per_page: Dict[int, List[torch.Tensor]], pbar ):

    # === Load model ===
    model = pretrain.model.get_model()
    model_weights = torch.load( model_path, map_location=torch.device( config.device ))
    model.load_state_dict( model_weights )
    model.zero_grad()
    model.eval()
    model.to( config.device )

    # === Compute losses ===
    losses_per_page = {}
    for page, batches in batches_per_page.items():
        losses_per_page[page] = []
        for batch in batches:
            with torch.no_grad():
                try:
                    inputs = batch.to(config.device)
                    outputs = model(inputs, labels=inputs)
                    loss = outputs.loss.detach().item()
                    losses_per_page[page].append(loss)
                    pbar.set_description(f"Loss: {uid} - {loss}")
                except Exception as e:
                    bt.logging.error(f"Exception is here! error {traceback.print(e)}")
                    losses_per_page[page].append(math.inf)
    return losses_per_page
    

def run_step( wins_per_epoch, metagraph, global_step ):
    """
        Executes a single validation step.
        - 1. Generates random pages from Falcon Refined web for evaluation.
        - 2. Identifies available UIDs for model updating (uids must be serving and have a recent weight set event.)
        - 3. Computes losses for each batch and each UID to attain losses per batch.
        - 4. Determines the winning UID based on lowest average loss per batch.
        - 5. Logs win percentages for each UID to wandb and screen.
        - 6. Logs step results and updates weights and biases (wandb) if configured.

        Parameters:
        - wins_per_epoch (dict): A dictionary to record the number of wins per UID.
        - metagraph (object): An object representing the meta information of models.
        - wandb_step (int): The current step number for logging in weights and biases.

    """

    # Update all models from wandb runs and return a list of uids
    # their paths and timestamps.
    uids, paths, model_timestamps = get_or_update_model_info( metagraph )

    # Get next batches
    pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range(pretrain.n_eval_pages)]
    batches_per_page = { page: None for page in pages }
    for page in pages:
        batches_per_page[page] = list(pretrain.dataset.SubsetFalconLoader(
            batch_size=pretrain.batch_size,
            sequence_length=pretrain.sequence_length,
            pages = [ page ]
        ))
    
    # Compute losses per page
    losses_per_page_per_uid = { uid: None for uid in uids }
    pbar = tqdm( uids, desc="Loss:", leave=False)
    for uid_i in pbar:
        model_path = paths[ uid ]
        losses_per_page_per_uid[ uid_i ] = compute_losses_per_page( uid, model_path, pages, batches_per_page )

    # Compute average loss per page
    average_loss_per_uid_per_page = { uid: {} for uid in uids }
    for uid_i in uids:
        for page_j in pages:
            losses = losses_per_page_per_uid[ uid_i ][ page_j ]
            average_loss = sum(losses)/len(losses)
            average_loss_per_uid_per_page[ uid_i ][ page_j ] = average_loss

    # Compute best average loss
    best_average_loss = math.inf
    best_average_loss_uid = None
    for uid_i in uids:
        average_loss = sum([ average_loss_per_uid_per_page[uid_i][page_j] for page_j in pages ] ) / len( pages )
        if average_loss < best_average_loss:
            best_average_loss = average_loss
            best_average_loss_uid = uid_i

    # Function returns True if this uid has lowest loss across all other uids on this 
    # batch, in case of ties takes uid with better timestamp.
    def is_win_loss_with_timestamps( this_uid, page_j, batch_k ):
        this_loss = losses_per_page_per_uid[ this_uid ][page_j][batch_k]
        this_timestamp = model_timestamps[ this_uid ]
        for other_uid in uids:
            other_loss = losses_per_page_per_uid[ other_uid ][ page_j ][ batch_k ]
            other_timestamp = model_timestamps[ other_uid ]
            if this_loss > other_loss:
                return False
            elif this_loss == other_loss and this_timestamp > other_timestamp:
                return False
        return True

    # Compute total wins per uid per page 
    total_wins_per_uid_per_page = { uid: { page: 0 for page in pages } for uid in uids }
    for uid in uids:
        for page in pages:
            for batch, _ in enumerate( batches_per_page[page] ):
                if is_win_loss_with_timestamps( uid, page, batch ):
                    total_wins_per_uid_per_page[ uid ][ page ] += 1
                    wins_per_epoch[ uid ] += 1 if uid in wins_per_epoch else 0

    # Build step log
    step_log = {
        'timestamp': time.time(),
        'pages': pages,
        'uids': uids,
        'best_average_loss': best_average_loss,
        'best_average_loss_uid': best_average_loss_uid
    }
    for uid in uids:
        uid_log = {
            'uid': uid,
            'timestamp': model_timestamps[ uid ],
        }
        for page in pages:
            uid_log[ page ] = {
                'page': page,
                'losses': losses_per_page_per_uid[ uid ][ page ],
                'average_loss': average_loss_per_uid_per_page[ uid ][page],
                'wins': total_wins_per_uid_per_page[ uid ][ page ],
                'win_rate': total_wins_per_uid_per_page[ uid ][ page ] / batches_per_page[ page ]
            }
        step_log[ str(uid) ] = uid_log

    # Sink step log.
    bt.logging.success(f"Step results: {step_log}")
    with open ( config.full_path + "/step_results.json", "a") as f:
        json.dump(step_log, f)
    if config.wandb.on: wandb.log( step_log, step = global_step )

def run_epoch( wins_per_epoch, global_step ):
    """
        Completes the validation epoch determining the weights that need to be set on chain
        and firing off the extrinsic. 

        Parameters:
        - wins_per_epoch (dict): A dictionary to record the number of wins per UID.
        - wandb_step (int): The current step number for logging in weights and biases.
    """
    # === Compute weights from wins ===
    weights = torch.zeros( len(metagraph.hotkeys) )
    for uid in wins_per_epoch:
        weights[uid] = wins_per_epoch[uid] / sum( wins_per_epoch.values() )
        if config.wandb.on: wandb.log( {f"wins_per_epoch/{uid}": wins_per_epoch[uid]/ sum(wins_per_epoch.values())}, step = global_step )
    wins_per_epoch = {}

    # === Set weights ===
    subtensor.set_weights(
        netuid = pretrain.NETUID,
        wallet = wallet,
        uids = metagraph.uids,
        weights = weights,
        wait_for_inclusion=False,
    )
    bt.logging.success(f"Set weights: {weights.tolist()}")

# === Validating loop ===
epoch_step = 0 
global_step = 0
last_epoch = metagraph.block.item()
bt.logging.success(f"Starting validator loop")
while True:
    try:
        # Init a new dict for counters.
        wins_per_epoch = {}
    
        while metagraph.block.item() - last_epoch < config.blocks_per_epoch:
            # Updates models (if needed) runs an eval step over each model
            # Records the number of 'wins' per model in the step. A wins
            # is defined as the model with the lowest loss on a given batch.
            run_step( wins_per_epoch, metagraph, global_step )
            metagraph = subtensor.metagraph( pretrain.NETUID )
            bt.logging.success(f"{metagraph.block.item() - last_epoch } / {config.blocks_per_epoch} blocks until next epoch.")
            global_step += 1

        # Finish epoch.
        run_epoch( wins_per_epoch, global_step )
        last_epoch = metagraph.block.item()
        epoch_step += 1

    except KeyboardInterrupt:
        bt.logging.info("KeyboardInterrupt caught, gracefully closing the wandb run...")
        if config.wandb.on: wandb_run.finish()
        exit()

    except Exception as e:
        bt.logging.error(f"Error in validator loop \n {e} \n {traceback.format_exc()}")


