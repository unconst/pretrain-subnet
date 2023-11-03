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

import torch
import random
import pretrain
import transformers
import argparse
import bittensor as bt
import requests
from transformers import AutoModelForPreTraining, AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoModel


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

    # Pull random batches from Falcon Dataset
    random_pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages)]
    loader = pretrain.dataset.SubsetFalconLoader(
        batch_size = 3,
        sequence_length = 512,
        pages = random_pages
    )
    data_list = list(loader)

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

        loss_dict[uid] = {'loss': None, 'timestamp': None, 'huggingface_repo': None, 'hotkey': None }
        axon = metagraph.axons[uid]
        response = dendrite.query(axon, pretrain.protocol.GetRepo(), timeout=0.5)
        if not response.is_success:
            bt.logging.info(f"failed response from uid {uid}")
            continue

        huggingface_repo = response.huggingface_repo
        bt.logging.info(f"got repo {huggingface_repo} from uid {uid}")
        loss_dict[uid]["huggingface_repo"] = huggingface_repo
        hotkey = huggingface_repo.split('/')[-1]

        if hotkey != metagraph.hotkeys[uid]:
            raise ValueError("Hotkey mismatch")
        
        try:
            filename = "model.safetensors"
            bt.logging.info(f"downloading model from {huggingface_repo}")
            config = AutoConfig.from_pretrained(huggingface_repo)
            model = AutoModelForCausalLM.from_pretrained(huggingface_repo)
            tokenizer = AutoTokenizer.from_pretrained(huggingface_repo)
            repo_api_url = f"https://huggingface.co/api/repos/{hotkey}"
            response = requests.get(repo_api_url)

            if response.ok:
                repo_info = response.json()
                timestamp = repo_info.get('lastModified', None)
                loss_dict[uid]["timestamp"] = timestamp
            else:
                bt.logging.error(f"Failed to get repo info from {repo_api_url}")

        except Exception as e:
            bt.logging.error(f"Error in downloading weights of uid {uid} \n {e}")
            continue
            
        model.zero_grad()
        model.train()
        bt.logging.info(f"starting eval loop on uid {uid}")
        average_loss = 0
        num_batches = 0

        for i, batch in enumerate(data_list):
            try:
                bt.logging.info(f"starting batch {i}")
                
                # Ensure batch is a list of strings
                if not all(isinstance(text, str) for text in batch):
                    raise ValueError("Batch must be a list of strings.")
                
                inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True)
                inputs = {k: v.to(config.device) for k, v in inputs.items()}  # Send inputs to device
                
                # Forward pass, get the outputs from the model
                outputs = model(**inputs, labels=inputs['input_ids'])
                
                # Extract the loss
                loss = outputs.loss.item()  # Use `.item()` to get a Python float
                average_loss += loss
                num_batches += 1
                bt.logging.info(f'Batch {i} loss: {loss}')  # Changed from success to info

            except Exception as e:
                bt.logging.error(f"Error in loss calc of uid {uid} \n {e}")
                continue 

        average_loss /= max(num_batches, 1)
        bt.logging.info(f"average_loss = {average_loss}")
        previous_loss = loss_dict[uid]["loss"]

        if previous_loss == None:
            loss_dict[uid]["loss"] = average_loss
        else:
            if average_loss < previous_loss:
                # Update dict with better loss and timestamp
                loss_dict[uid]["loss"] = average_loss

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

    bt.logging.info(f"setting weights... Scores = {weights}")
    subtensor.set_weights(
        netuid = pretrain.NETUID,
        wallet = wallet,
        uids = metagraph.uids,
        weights = weights,
        wait_for_inclusion=False,
    )
    


