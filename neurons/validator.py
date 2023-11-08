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

# === Init vars ===
api = wandb.Api( timeout = 100 )
wins_per_epoch = {}
model_paths = {}
model_timestamps = {}

def update_models( log, valid_runs ):
    pbar = tqdm( list(valid_runs.items()) , desc="Updating models:", leave=False )
    for hotkey, run_info in pbar:

        pbar.set_description(f"Updating models: {run_info['run']}")
        # Get run info.
        log[str(uid)]['run_id'] = run_info['run']
        uid = run_info['uid']
        model_file = run_info['model_artifact']
        model_timestamp = run_info['timestamp']
        model_dir = f'{config.full_path}/models/{hotkey}/'
        timestamp_file = f'{model_dir}timestamp.json'
        model_paths[uid] = model_dir + ARTIFACT_NAME

        # Check if the timestamp file exists and if the timestamp matches ===
        if os.path.exists(timestamp_file):
            with open(timestamp_file, 'r') as f:
                existing_timestamp = json.load(f)
            if existing_timestamp == model_timestamp:
                continue

        # If the timestamp does not match or the file does not exist, update the timestamp ===
        os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists
        with open(timestamp_file, 'w') as f:
            json.dump(model_timestamp, f)

        # Replace model.
        model_file.download(replace=True, root=model_dir)


def compute_losses_on_batches( uid, eval_batches: Dict[int, List[torch.Tensor]], device, pbar, log, random_pages ):

    """ Computes the average loss of a model on a list of batches.
        Args:
            batches (:obj:`List[torch.Tensor]`): The batches to evaluate on.
            device (:obj:`torch.device`): The device to evaluate on.
    """

    # === Load model ===
    model = pretrain.model.get_model()
    model_weights = torch.load( model_paths[uid], map_location=torch.device(device))
    model.load_state_dict( model_weights )
    model.zero_grad()
    model.eval()
    model.to( device )

    # === Compute losses ===
    for page, batches in eval_batches.items():
        if page not in log[str(uid)]:
            log[str(uid)][page] = {}
        log[str(uid)][page]["losses"] = []
        losses = log[str(uid)][page]["losses"]
        for batch in batches:
            with torch.no_grad():
                try:
                    inputs = batch.to(device)
                    outputs = model(inputs, labels=inputs)
                    loss = outputs.loss.detach().item()
                    losses.append(loss)
                    pbar.set_description(f"Loss: {uid} - {loss}")
                except Exception as e:
                    bt.logging.error(f"Exception is here! error {traceback.print(exc)}")
                    losses.append(math.inf)

def run_step( wins_per_epoch, metagraph, wandb_step ):
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
    # === Get next batches ===
    random_pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range(pretrain.n_eval_pages)]
    bt.logging.info(f"using random pages: {random_pages}")

    # Initialize an empty dictionary for eval_batches
    eval_batches = {}

    # For each unique page, create a list of batches
    total_batches = 0
    for page in set(random_pages):
        eval_batches[page] = list(pretrain.dataset.SubsetFalconLoader(
            batch_size=pretrain.batch_size,
            sequence_length=pretrain.sequence_length,
            pages=[page]
        ))
        total_batches += len( eval_batches )

    # Update model for each uid ===
    valid_runs = pretrain.get_valid_runs( metagraph )
    valid_uids = [ v['uid'] for v in valid_runs.values() ]
    log = { str(uid): {} for uid in valid_uids }

    # Update all models if need be.
    update_models( log, valid_runs )

    # Compute losses on each batch ===
    best_uid = None
    best_average_loss = 1000

    # Compute losses per batch per page.
    pbar = tqdm(valid_uids, desc="Loss:", leave=False)
    for uid in pbar:
        compute_losses_on_batches(uid, eval_batches, config.device, pbar, log, random_pages)
        for page in random_pages:
            losses = log[str(uid)][page]["losses"]
            if math.inf not in losses:
                average_loss = sum(losses) / len(losses)
                log[str(uid)][page]["average_loss"] = average_loss
                if average_loss < best_average_loss:
                    best_average_loss = average_loss
                    best_uid = uid
    if best_uid != None:
        log["best_average_loss"] = best_average_loss
        log["best_average_loss_uid"] = best_uid 
        log["pages"] = random_pages
        log["timestamp"] = time.time()

    # Compute wins per batch per page.
    for uid_i in valid_uids:
        for page in random_pages:
            for loss_i in log[str(uid_i)][page]["losses"]:
                is_win = True
                for uid_j in valid_uids:
                    loss_j = log[str(uid_j)][page]["losses"]
                    if loss_j < loss_i and model_timestamps[ uid_i ] < model_timestamps[ uid_j ]:
                        is_win = False
                        break
                log[str(uid_i)][page]["wins"].append( is_win )
    
    # Compute wins per step
    for uid_i in valid_uids:
        total_wins = 0
        for page in random_pages:
            total_wins += sum( log[str(uid_i)][page]["wins"] ) 
        log[str(uid)]["total_wins"] = total_wins
        log[str(uid)]["win percentage"] = total_wins / total_batches

    # Clear uid logs for empty dictionaries.
    for key in list(log): 
        if log[key] == {}: del log[key]
    bt.logging.success(f"Step results: {log}")
    with open ( config.full_path + "/step_results.json", "a") as f:
        json.dump(log, f)
    if config.wandb.on: wandb.log( log, step = wandb_step )

def run_epoch( wins_per_epoch, wandb_step ):
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
        if config.wandb.on: wandb.log( {f"wins_per_epoch/{uid}": wins_per_epoch[uid]/ sum(wins_per_epoch.values())}, step = wandb_step )
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
step = 0
epoch = 0 
last_epoch = metagraph.block.item()
bt.logging.success(f"Starting validator loop")
while True:
    try:
        while metagraph.block.item() - last_epoch < config.blocks_per_epoch:
            # Updates models (if needed) runs an eval step over each model
            # Records the number of 'wins' per model in the step. A wins
            # is defined as the model with the lowest loss on a given batch.
            run_step( wins_per_epoch, metagraph, step )
            metagraph = subtensor.metagraph( pretrain.NETUID )
            bt.logging.success(f"{metagraph.block.item() - last_epoch } / {config.blocks_per_epoch} blocks until next epoch.")
            step += 1

        # Finish epoch.
        run_epoch( wins_per_epoch, step )
        last_epoch = metagraph.block.item()
        epoch += 1

    except KeyboardInterrupt:
        bt.logging.info("KeyboardInterrupt caught, gracefully closing the wandb run...")
        if config.wandb.on: wandb_run.finish()
        exit()

    except Exception as e:
        bt.logging.error(f"Error in validator loop \n {e} \n {traceback.format_exc()}")


