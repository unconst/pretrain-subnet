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
import json
import math
import time
import wandb
import torch
import string
import random
import pretrain
import traceback
import argparse
import bittensor as bt
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List

# Global artifact name
ARTIFACT_NAME:str = "model.pth"

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")
    parser.add_argument( '--wandb.off', dest = 'wandb.on', action='store_false', help='Turn off wandb logging.' )
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
    config.type = "validator"
    wandb_run =  wandb.init(
        name = run_name,
        anonymous = "allow",
        reinit = False,
        project = 'openpretraining',
        entity = 'opentensor-dev',
        config = config,
        dir = config.full_path,
    )
    # Sign wandb run.
    wandb.init( project = "openpretraining", entity="opentensor-dev" )
    config.signature = wallet.hotkey.sign( wandb_run.id.encode() ).hex()
    wandb.config.update( config, allow_val_change=True )

def update_models(metagraph, blacklisted_models):
    """
        This function updates local model files by synchronizing with models registered in the metagraph
        and available on Weights & Biases (wandb). It checks the legitimacy of each model's signature,
        downloads models that are either new or updated, and keeps track of their timestamps and paths.
        Args:
            metagraph (Metagraph): An object representing the metagraph of the models to be synchronized.
    """
    
    # Initialize Weights & Biases API client with a timeout setting
    api = wandb.Api(timeout=100)
    # Retrieve runs from the wandb project
    runs = api.runs(
        "opentensor-dev/openpretraining",
        filters = { 
            "config.version": pretrain.__version__,
            "config.type": "miner",
            "config.hotkey": {"$regex": "^.+$"}
        } 
    )
    bt.logging.trace( f'Got runs: {[r for r in runs]}' )

    # Use tqdm to show progress bar for the iteration over runs
    pbar = tqdm(runs, desc="Getting runs:", leave=False)

    # Iterate over each run in the project
    has_updated = {}
    for run in pbar:
        try:
            pbar.set_description(f"Updating: {run.id}")
            bt.logging.trace(f'Updating: {run.id}')

            # Skip models that are not registered in the metagraph
            hotkey = run.config['hotkey']
            if hotkey not in metagraph.hotkeys: continue
            uid = metagraph.hotkeys.index(hotkey)
            bt.logging.trace(f'uid: {uid}')

            # Ensure a 'signature' exists and verify its legitimacy
            if 'signature' not in run.config: continue
            signature = run.config['signature']
            keypair = bt.Keypair(ss58_address=hotkey)
            if not keypair.verify(run.id, bytes.fromhex(signature)): continue

            # Check if we have updated this uid already:
            if uid in has_updated: 
                bt.logging.trace(f'already updated this uid')
                continue
            else: has_updated[uid] = True

            # Attempt to access the model artifact file
            try: model_artifact = run.file('model.pth')
            except: continue
            bt.logging.trace(f'model_artifact: {model_artifact}')

            # Convert the updatedAt string to a timestamp
            try: remote_model_timestamp = int(datetime.strptime(model_artifact.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
            except: continue

            # Define the local model directory and timestamp file paths
            model_dir = os.path.join(config.full_path, 'models', str(uid))
            metadata_file = os.path.join(model_dir, 'metadata.json')
            model_path = os.path.join(model_dir, 'model.pth')
            bt.logging.trace(f'model_dir: {model_dir}')
            bt.logging.trace(f'metadata_file: {metadata_file}')
            bt.logging.trace(f'model_path: {model_path}')

            # Function to determine if the model needs updating
            def needs_updating():
                # Check if we can load the local model.
                try:
                    torch.load( model_path )
                except:
                    bt.logging.trace(f'Model path corrupted, needs redownloading with path: {model_path}') 
                    return True
                # Check if we have a timestamp file.
                if not os.path.exists(metadata_file):
                    bt.logging.trace(f'No timestamp, needs downloading with path') 
                    return True
                # Check if the local timestamp is older.
                with open(metadata_file, 'r') as f:
                    existing_timestamp = json.load(f)['timestamp']
                # Check timestamp.
                if existing_timestamp < remote_model_timestamp:
                    bt.logging.trace(f'Existing timestamp: {existing_timestamp} is older than newer timestamp: {remote_model_timestamp}') 
                    return True
                else:
                    bt.logging.trace(f'Existing timestamp: {existing_timestamp} is newer than older timestamp: {remote_model_timestamp}') 
                    return False

            # Function to update the model file and its timestamp
            if needs_updating():
                os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists
                with open(metadata_file, 'w') as f: 
                    json.dump( 
                        { 
                            'timestamp': remote_model_timestamp, 
                            'runid': run.id,
                            'model_path': model_path,
                            'version': run.config['version']
                        }, f)
                if uid in blacklisted_models:
                    blacklisted_models.remove(uid)
                model_artifact.download(replace=True, root=model_dir)
                bt.logging.debug( f'Updated model under path: { model_dir } with timestamp: { remote_model_timestamp }')
            else:
                bt.logging.trace( f'Didnt update model under path: { model_dir } with timestamp: { remote_model_timestamp }')

        except Exception as e:
            bt.logging.error(f'Error updating run with error: {e}')
            continue

def get_uid_metadata(metagraph):
    """
        Retrieves the file paths to model checkpoints and their associated timestamps for each UID in the metagraph.

    Args:
        metagraph (object): An object containing UIDs for models.
    
    Returns:
        tuple: A tuple containing a list of UIDs that were successfully processed, 
            a dictionary mapping UIDs to their model file paths, 
            and a dictionary mapping UIDs to their timestamp data.
    """
    # Initialize dictionaries for model paths and timestamps.
    metadata = {}
    # Iterate over each UID in the metagraph.
    for uid in metagraph.uids.tolist():
        try:
            # Fill metadata from files and check if we can load weights.
            model_dir = os.path.join(config.full_path, 'models', str(uid))
            try:
                model_path = os.path.join(model_dir, 'model.pth')
                model_weights = torch.load( model_path )
            except Exception as e:
                bt.logging.trace(f'Cant load weights under: {model_path}')
                continue
            try:
                model = pretrain.model.get_model()
                model.load_state_dict(model_weights)
            except Exception as e:
                bt.logging.trace(f'Cant load weights into model')
                continue
            try:
                with open(os.path.join(model_dir, 'metadata.json'), 'r') as f: 
                    meta = json.load(f)
            except Exception as e:
                bt.logging.trace(f'Cant load metadata from json.')
                continue
            if 'version' not in meta or 'timestamp' not in meta or 'runid' not in meta:
                bt.logging.trace(f'metadata is malformed: {meta}')
                continue
            if meta['version'] != pretrain.__version__:
                version = meta['version']
                bt.logging.trace(f'Model verison is out of date, with version: {version}')
                continue
            else:
                # Valid metadata.
                metadata[uid] = meta
        except Exception as e:
            print (e)
            # Skip this UID if any error occurs during loading of model or timestamp.
            continue
    # Return metadata.
    return metadata


def compute_losses_per_page(uid: int, model_path: str, batches_per_page: Dict[int, List[torch.Tensor]], pbar=None) -> Dict[int, List[float]]:
    """
    Computes the loss for each page of batches using the given pre-trained model.
    
    Args:
        uid (int): The unique identifier for the current model.
        model_path (str): The file path to the pre-trained model.
        batches_per_page (Dict[int, List[torch.Tensor]]): A dictionary where each key is a page number
            and each value is a list of batch tensors to be processed by the model.
        pbar: A tqdm progress bar instance for real-time progress updates.
    
    Returns:
        Dict[int, List[float]]: A dictionary mapping each page to a list of loss values for the batches on that page.
    """
    bt.logging.trace( f'Computing loss for uid: {uid} on model path: {model_path} for page batches: {list(batches_per_page.keys())}')

    try:
        # Load the pre-trained model from the specified path
        model = pretrain.model.get_model()
        model_weights = torch.load(model_path, map_location=torch.device(config.device))
        model.load_state_dict(model_weights)
        model.eval()  # Set the model to evaluation mode
        model.to(config.device)  # Move the model to the appropriate device
    except Exception as e:
        bt.logging.debug(f"Error loading model under path {model_path} with error: {e}")
        inf_losses = {}
        for page, batches in batches_per_page.items():
            inf_losses[page] = [math.inf for _ in batches]
        return inf_losses

    # Initialize a dictionary to store loss values for each page
    losses_per_page = {}

    # Iterate over each page and its corresponding batches
    for page, batches in batches_per_page.items():
        page_losses = []  # List to store losses for the current page

        # Process each batch and compute its loss
        for batch in batches:
            try:
                # Perform a forward pass with the model to compute the loss
                inputs = batch.to(config.device)
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss.item()  # Get the scalar loss value
                page_losses.append(loss)
                if pbar is not None:
                    pbar.set_description(f"Loss: {uid} - {loss:.4f}")
            except Exception as e:
                # Log the exception and append infinity to indicate failure
                bt.logging.error(f"Exception occurred: {e}")
                traceback.print_exc()  # Correctly print the stack trace
                page_losses.append(math.inf)
        
        # Update the dictionary with the losses for the current page
        losses_per_page[page] = page_losses

    return losses_per_page
    

def run_step( wins_per_epoch, losses_per_epoch, global_best_uid, metagraph, global_step, blacklisted_models ):
    """
        Executes a single validation step.
        - 1. Updates local models using wandb data.
        - 2. Generates random pages from Falcon Refined web for evaluation.
        - 3. Computes losses for each batch and each UID to attain losses per batch per page
        - 4. Determines the winning UID based on lowest average loss per batch.
        - 5. Logs win percentages for each UID to wandb and screen.
        - 6. Logs step results and updates weights and biases (wandb) if configured.

        Parameters:
        - wins_per_epoch (dict): A dictionary to record the number of wins per UID.
        - losses_per_epoch (dict): A dictionary to record all losses in epoch.
        - global_best_uid (int): Uid with lowest global loss at beginning of epoch.
        - metagraph (object): An object representing the meta information of models.
        - wandb_step (int): The current step number for logging in weights and biases.

    """

    # Update all models from wandb runs and return a list of uids
    # their paths and timestamps.
    metadata = get_uid_metadata( metagraph )
    uids = [uid for uid in metadata if uid not in blacklisted_models]
    bt.logging.debug( f'Runnning step with uids: {uids}, metadata: {metadata}')

    # Generate random pages for evaluation and prepare batches for each page
    pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range(pretrain.n_eval_pages)]

    total_batches = sum([ len(b) for b in batches_per_page.values()] )
    
    # Compute losses per page
    losses_per_page_per_uid = { uid: None for uid in uids }
    pbar = tqdm( uids, desc="Loss:", leave=False)
    for uid_i in pbar:
        model_path = metadata[ uid_i ]['model_path']
        losses  = compute_losses_per_page( uid_i, model_path, batches_per_page, pbar )
        losses_per_page_per_uid[ uid_i ] = losses

    # Compute average loss per page
    average_loss_per_uid_per_page = { uid: {} for uid in uids }
    for uid_i in uids:
        for page_j in pages:
            losses = losses_per_page_per_uid[ uid_i ][ page_j ]
            average_loss = sum(losses)/len(losses)
            average_loss_per_uid_per_page[ uid_i ][ page_j ] = average_loss
            if uid_i in losses_per_epoch:
                losses_per_epoch[ uid_i ].append(average_loss)
            else:
                losses_per_epoch[ uid_i ] = [average_loss]


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
    def is_winning_loss_with_timestamps( this_uid, page_j, batch_k ):
        this_loss = losses_per_page_per_uid[ this_uid ][page_j][batch_k]
        if this_uid == global_best_uid:
            this_loss *= (1 - pretrain.per_loss_epsilon)
        this_timestamp = metadata[ this_uid ]['timestamp']
        for other_uid in uids:
            other_loss = losses_per_page_per_uid[ other_uid ][ page_j ][ batch_k ]
            other_timestamp = metadata[ other_uid ]['timestamp']
            if this_loss > other_loss:
                return False
            elif this_loss == other_loss and this_timestamp > other_timestamp:
                return False
        return True

    # Compute total wins per uid per page 
    total_wins_per_uid_per_page = { uid: { page: 0 for page in pages } for uid in uids }
    for uid in uids:
        wins_per_epoch[ uid ] = 0
        for page in pages:
            for batch, _ in enumerate( batches_per_page[page] ):
                if is_winning_loss_with_timestamps( uid, page, batch ):
                    total_wins_per_uid_per_page[ uid ][ page ] += 1
                    wins_per_epoch[ uid ] += 1 
        if wins_per_epoch[ uid ] == 0:
            blacklisted_models.append(uid)

    bt.logging.debug(f"adding {blacklisted_models} uids to ")
    # Build step log
    step_log = {
        'timestamp': time.time(),
        'pages': pages,
        'uids': uids,
        'best_average_loss': best_average_loss,
        'best_average_loss_uid': best_average_loss_uid,
        'uid_data': {}
    }
    for uid in uids:
        average_losses = [average_loss_per_uid_per_page[uid][pagek] for pagek in pages]
        average_loss = sum(average_losses) / len(average_losses)
        win_rate = sum ( [total_wins_per_uid_per_page[ uid ][ pagek ] for pagek in pages ]) / total_batches
        win_total = sum ( [total_wins_per_uid_per_page[ uid ][ pagek ] for pagek in pages ])
        step_log['uid_data'][ str(uid) ] = {
            'uid': uid,
            'runid': metadata[ uid ]['runid'],
            'timestamp': metadata[ uid ]['timestamp'],
            'average_losses': average_losses,
            'average_loss': average_loss,
            'win_rate': win_rate,
            'win_total': win_total,
        }

    # Sink step log.
    bt.logging.success(f"Step results: {step_log}")
    original_format_json = json.dumps(step_log)
    uids = step_log['uids']
    uid_data = step_log['uid_data']

    # Create a new dictionary with the required format
    graphed_data = {
        'uid_data': {str(uid): uid_data[str(uid)]['average_loss'] for uid in uids}
    }
    if config.wandb.on: wandb.log({ **graphed_data, "original_format_json": original_format_json}, step=global_step)

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
        weights[uid] = (wins_per_epoch[uid] + 1)/ sum( wins_per_epoch.values() )
    bt.logging.debug(f"wins_per_epoch = {wins_per_epoch} ------- global best = {global_best_uid}: {round(global_best_loss, 4)}")
    wins_per_epoch = {}

    # === Set weights ===
    subtensor.set_weights(
        netuid = pretrain.NETUID,
        wallet = wallet,
        uids = metagraph.uids,
        weights = weights,
        wait_for_inclusion=False,
    )
    bt.logging.success(f"Set weights successfully")
    bt.logging.debug(f"Weights info: {weights.tolist()}")

def get_best_uid():
    global_best_uid = max(get_uid_metadata(metagraph), key=lambda uid: metagraph.I[uid].item())
    bt.logging.info(f"initial global best uid is {global_best_uid}")
    losses_dict = compute_losses_per_page(global_best_uid, get_uid_metadata( metagraph )[ global_best_uid ]['model_path'], batches_per_page)
    total_losses = [value for values in losses_dict.values() for value in values]
    global_best_loss = sum(total_values) / len(total_values)
    bt.logging.info(f"initial global best loss is {global_best_loss}")
    return global_best_uid, global_best_loss


# === Validating loop ===
epoch_step = 0 
global_step = 0
last_epoch = metagraph.block.item()
bt.logging.success(f"Starting validator loop")
pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range(pretrain.n_eval_pages)]
batches_per_page = {
        page: list(pretrain.dataset.SubsetFalconLoader(
            batch_size=pretrain.batch_size,
            sequence_length=pretrain.sequence_length,
            pages=[page]
        )) for page in pages
    }
total_batches = sum([ len(b) for b in batches_per_page.values()] )
while True:
    try:
        # Init a new dict for counters.
        wins_per_epoch = {}
        losses_per_epoch = {}
        blacklisted_models = []
        # Update all local models at beginning of epoch.
        metagraph = subtensor.metagraph( pretrain.NETUID )
        # update_models( metagraph, blacklisted_models )
        global_best_uid, global_best_loss = get_best_uid()
    
        while metagraph.block.item() - last_epoch < config.blocks_per_epoch:
            # Updates models (if needed) runs an eval step over each model
            # Records the number of 'wins' per model in the step. A wins
            # is defined as the model with the lowest loss on a given batch.
            run_step( wins_per_epoch, losses_per_epoch, global_best_uid, metagraph, global_step, blacklisted_models )
            metagraph = subtensor.metagraph( pretrain.NETUID )
            bt.logging.debug(f"{metagraph.block.item() - last_epoch } / {config.blocks_per_epoch} blocks until next epoch.")
            global_step += 1

        # Update global best loss and uid.
        for uid in losses_per_epoch.keys():
            epoch_average_loss = sum(losses_per_epoch[uid])/len(losses_per_epoch[uid])
            if epoch_average_loss < global_best_loss * (1 - pretrain.best_uid_epsilon):
                global_best_uid = uid
                global_best_loss = epoch_average_loss
        if config.wandb.on: wandb.log( {"global_best_uid": global_best_uid, "global_best_loss":global_best_loss }, step = global_step )

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


