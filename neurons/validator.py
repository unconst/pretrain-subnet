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
from tqdm import tqdm
from datetime import datetime

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
wins_per_epoch = {}
model_paths = {}
model_timestamps = {}

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


def compute_losses_on_batches( uid, batches: typing.List[torch.Tensor], device ):
    """ Computes the average loss of a model on a list of batches.
        Args:
            batches (:obj:`List[torch.Tensor]`): The batches to evaluate on.
            device (:obj:`torch.device`): The device to evaluate on.
    """
    # No model for this uid.
    if uid not in model_paths: 
        return [math.inf for _ in range(len(batches))]

    # === Load model ===
    model = pretrain.model.get_model()
    model_weights = torch.load( model_paths[uid], map_location=torch.device(device))
    model.load_state_dict( model_weights )
    model.zero_grad()
    model.eval()
    model.to( device )

    # === Compute losses ===
    losses_per_batch = []
    for i, batch in enumerate( batches ):
        with torch.no_grad():
            try:
                inputs = batch.to(model.device)
                outputs = model(inputs, labels=inputs)
                losses_per_batch.append( outputs.loss.detach().item() )
            except Exception as e:
                losses_per_batch.append( math.inf )
    return losses_per_batch

def optionally_update_model( uid: int ) -> pretrain.model.GPT2LMHeadModel:

    # == Get uid's run ==
    response = dendrite.query( metagraph.axons[uid], pretrain.protocol.GetRun(), timeout=0.5 )
    if not response.is_success: bt.logging.debug('Failed to get miner run'); return
    
    # === Get model run === 
    run_id = response.run_id
    run = api.run(f"opentensor-dev/openpretraining/{run_id}")
    if run == None: bt.logging.debug('Failed to get miner run'); return

    # === Check hotkey match ===
    hotkey = run.config.get('hotkey')
    if hotkey != metagraph.hotkeys[uid]: bt.logging.debug('Hotkey mismatch'); return 

    # === Check if model exist ===
    model_file = run.file( ARTIFACT_NAME )
    if model_file == None: bt.logging.debug('Miner has no model artifact.'); return 
    
    # === Check if the model needs updating ===    
    model_timestamp = int(datetime.strptime(model_file.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
    if uid in model_timestamps and model_timestamp == model_timestamps[uid]: bt.logging.debug('Miner model artifact is up to date.'); return 
    model_timestamps[uid] = model_timestamp # Update model timestamp.

    # === Load the model from file ===
    bt.logging.debug(f"Updating model for: {uid}")
    model_dir = f'{config.full_path}/models/{metagraph.hotkeys[uid]}/' 
    model_file.download( replace = True, root = model_dir) 
    model_paths[uid] = model_dir + ARTIFACT_NAME


def run_step( wins_per_epoch, metagraph ):
    # === Get next batches ===
    random_pages = [ random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range(pretrain.n_eval_pages) ]
    eval_batches = list(pretrain.dataset.SubsetFalconLoader(
        batch_size = pretrain.batch_size,
        sequence_length = pretrain.sequence_length,
        pages = random_pages
    ))

    # === UIDs to evaluate ===
    available = get_available_uids( metagraph ) 

    # === Update model for each uid ===
    for uid in tqdm( available , desc="Updating models", leave=False):
        optionally_update_model( uid )

    # === Compute losses on each batch ===
    best_uid = None
    best_average_loss = math.inf
    losses_per_uid_per_batch = {}
    average_loss_per_uid = {}
    for uid in tqdm(available, desc="Computing losses on batches", leave=False):
        losses_per_batch = compute_losses_on_batches( uid, eval_batches, config.device )
        losses_per_uid_per_batch[uid] = losses_per_batch
        if math.inf not in losses_per_batch:
            average_loss = sum(losses_per_batch) / len(losses_per_batch)
            average_loss_per_uid[uid] = average_loss
            if average_loss < best_average_loss: best_average_loss = average_loss; best_uid = uid
            if config.wandb.on and average_loss != math.inf: wandb.log( {f"average_loss/{uid}": average_loss} )
            bt.logging.success(f"Computed average_loss for uid: {uid} losses: {average_loss}")
    uid_losses = {f"{uid}": average_loss for uid, average_loss in average_loss_per_uid.items()}
    if config.wandb.on: wandb.log(uid_losses)
    if best_uid == None and config.wandb.on: wandb.log( {f"best_average_loss/{uid}": math.inf} )
    if best_uid == None and config.wandb.on: wandb.log( {f"best_average_loss_uid": best_uid} )

    # === Compute wins per batch ===
    win_per_step = {}
    for step in range(len(eval_batches)):
        min_loss = math.inf
        min_loss_uid = None
        for uid in losses_per_uid_per_batch:
            if losses_per_uid_per_batch[uid][step] < min_loss:
                min_loss = losses_per_uid_per_batch[uid][step]
                min_loss_uid = uid
        wins_per_epoch[min_loss_uid] = wins_per_epoch.get(min_loss_uid, 0) + 1
        win_per_step[min_loss_uid] = win_per_step.get(min_loss_uid, 0) + 1
    bt.logging.success(f"Computed wins per step: {win_per_step}")
    bt.logging.success(f"Computed wins per epoch: {wins_per_epoch}")

    # === Log wins per step ===
    for uid in win_per_step.keys():
        if config.wandb.on: wandb.log( {f"win_per_step/{uid}": win_per_step[uid] / (sum(win_per_step.values())) } )

def epoch( wins_per_epoch ):
    # === Compute weights from wins ===
    weights = torch.zeros( len(metagraph.hotkeys) )
    for uid in wins_per_epoch:
        weights[uid] = wins_per_epoch[uid] / sum( wins_per_epoch.values() )
        if config.wandb.on: wandb.log( {f"wins_per_epoch/{uid}": wins_per_epoch[uid]} )
    wins_per_epoch = {} # Clearn wins per epoch.

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
last_epoch = metagraph.block.item()
bt.logging.success(f"Starting validator loop")
while True:
    try:
        while metagraph.block.item() - last_epoch < config.blocks_per_epoch:
            # Updates models (if needed) runs an eval step over each model
            # Records the number of 'wins' per model in the step. A wins
            # is defined as the model with the lowest loss on a given batch.
            run_step( wins_per_epoch, metagraph )
            metagraph = subtensor.metagraph( pretrain.NETUID )
            bt.logging.success(f"{metagraph.block.item() - last_epoch } / {config.blocks_per_epoch} blocks until next epoch.")

        # Finish epoch.
        epoch( wins_per_epoch )
        last_epoch = metagraph.block.item()

    except KeyboardInterrupt:
        bt.logging.info("KeyboardInterrupt caught, gracefully closing the wandb run...")
        if config.wandb.on: wandb_run.finish()
        exit()

    except Exception as e:
        bt.logging.error(f"Error in validator loop \n {e} \n {traceback.format_exc()}")


