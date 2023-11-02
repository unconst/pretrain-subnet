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

import wandb
import torch
import random
import pretrain
import argparse
import bittensor as bt


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)
    return config

config = get_config()
bt.logging( config = config )
wallet = bt.wallet( config = config )
subtensor = bt.subtensor( config = config )
dendrite = bt.dendrite( wallet = wallet )
metagraph = subtensor.metagraph( pretrain.NETUID )
torch.backends.cudnn.benchmark = True
loss_dict = {}

while True:
    bt.logging.info("starting validator")
    api = wandb.Api( timeout = 100 )

    # Pull random batches from Falcon Dataset
    random_pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages)]
    loader = pretrain.dataset.SubsetFalconLoader(
        batch_size = 3,
        sequence_length = 512,
        pages = random_pages
    )

    # Init vars
    best_average_loss = None
    best_uid = None
    best_uid_run = None
    metagraph = subtensor.metagraph( pretrain.NETUID )
    uids = metagraph.uids

    # Iterate through all uids and evaluate the loss of their model weights against the random batches
    available_uids = [ uid.item() for uid in metagraph.uids if metagraph.axons[uid].is_serving and (metagraph.block.item() - metagraph.last_update[uid] < 500)]
    for uid in available_uids:
        bt.logging.info(f"starting loop on uid {uid}")

        loss_dict[uid] = {'loss': None, 'timestamp': None, 'run_id': None, 'hotkey': None }
        axon = metagraph.axons[uid]
        response = dendrite.query( axon, pretrain.protocol.GetRun(), timeout=1 )
        if not response.is_success:
            bt.logging.info(f"failed response from uid {uid}")
            continue

        run_id = response.run_id
        bt.logging.info(f"got run name {run_id} from uid {uid}")
        run = api.run(f"opentensor-dev/openpretraining/{run_id}")
        loss_dict[uid]["run_id"] = run

        # Hotkey of run must match that of the sending hotkey
        hotkey = run.config.get('hotkey')
        loss_dict[uid]["hotkey"] = hotkey
        if hotkey != metagraph.hotkeys[uid]:
            raise ValueError("Hotkey mismatch")
        
        # Download the model weights
        try:
            artifact_name = "model.pth"
            bt.logging.info(f"downloading weights from {artifact_name}")

            run.file(artifact_name).download(replace=True)

            model = pretrain.model.get_model()
            # model_weights = torch.load(artifact_name)
            # CPU option 
            model_weights = torch.load(artifact_name, map_location=torch.device('cpu'))

            model.load_state_dict(model_weights)
        except Exception as e:
            bt.logging.error(f"Error in downloading weights of uid {uid} \n {e}")
            continue
        model.zero_grad()
        model.train()
        model.to( config.device )

        # Run eval
        bt.logging.info(f"starting eval loop on uid {uid}")
        average_loss = 0
        num_batches = 0
        for i, batch in enumerate(loader):
            try:
                bt.logging.info(f"Processing batch {i}")
                inputs = batch.to(model.device)
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss.detach().item()
                average_loss += loss
                num_batches += 1
                bt.logging.success(f'Acc: step: {i} loss: {loss}')

            except Exception as e:
                bt.logging.error(f"Error in loss calc of uid {uid} \n {e}")

        average_loss /= max(num_batches, 1)
        bt.logging.info(f"average_loss = {average_loss}")
        previous_loss = loss_dict[uid]["loss"]

        if previous_loss == None:
            loss_dict[uid]["loss"] = average_loss
            loss_dict[uid]["timestamp"] = run.created_at
        else:
            if average_loss < previous_loss:
                # Update dict with better loss and timestamp
                loss_dict[uid]["loss"] = average_loss
                loss_dict[uid]["timestamp"] = run.created_at

    # Get best average loss and best uid
    # Best uid if tie on loss is based on timestamp of run upload
    best_average_loss = None
    best_timestamp = None
    best_uid = None
    for uid in loss_dict.keys():
        uid_loss = loss_dict[uid]['loss']
        uid_timestamp = loss_dict[uid]['timestamp']
        if uid_loss == None: continue
        if best_average_loss == None:
            best_average_loss = uid_loss
            best_uid = uid
            best_timestamp = uid_timestamp
        else:
            if uid_loss < best_average_loss:
                best_average_loss = uid_loss
                best_uid = uid
                best_timestamp = uid_timestamp
            elif uid_loss == best_average_loss:
                if uid_timestamp < best_timestamp:
                    best_average_loss = uid_loss
                    best_uid = uid
                    best_timestamp = uid_timestamp


    bt.logging.info(f"uid {best_uid} has  best loss of {best_average_loss} and timestamp {best_timestamp}")

    if best_uid != None:
        weights = torch.zeros_like( metagraph.S )
        weights[best_uid] = 1
    else:
        weights = torch.ones_like( metagraph.S )

    subtensor.set_weights(
        netuid = pretrain.NETUID,
        wallet = wallet,
        uids = metagraph.uids,
        weights = weights,
        wait_for_inclusion=False,
    )
    


