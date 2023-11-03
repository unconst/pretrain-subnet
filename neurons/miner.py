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

# pip install huggingface_hub
# install git-lfs
# huggingface-cli login with personal access token

import os
import time
import torch
import string
import random
import argparse
import pretrain
import traceback
import bittensor as bt
from huggingface_hub import HfApi, HfFolder, Repository

# === Config ===
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model_path", type = str, help="Run name.", default='~/model.pth' )
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)
    return config

config = get_config()

# === Bittensor objects ===
bt.logging( config = config )
bt.logging.success( config )
wallet = bt.wallet( config = config ) 
subtensor = bt.subtensor( config = config )
metagraph = subtensor.metagraph( pretrain.NETUID )
if wallet.hotkey.ss58_address not in metagraph.hotkeys: raise Exception("You are not registered. Use `btcli s recycle_register` to register.")
my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
bt.logging.success( f'You are registered with address: {wallet.hotkey.ss58_address} and uid: {my_uid}' )

# === Authenticate to Hugging Face ===
api = HfApi()
user = api.whoami(HfFolder.get_token())
username = user['name']

# === Prepare Hugging Face repository ===
model_name = f"{wallet.hotkey.ss58_address}"
repo_name = f"{username}/{model_name}"
repo_url = api.create_repo(repo_name, private=False, exist_ok=True) 
repo_local_path = os.path.expanduser(f"~/pretrain-subnet/neurons/pretraining_model/{model_name}")

repo = Repository(local_dir=repo_local_path, clone_from=repo_url)
bt.logging.info(f"Cloned repository {repo_name} to {repo_local_path}")


def update_model(model, model_path):
    # Using the Repository object to manage files and versions
    repo.git_add(model_path)
    repo.git_commit(f"Update model at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    repo.git_push()
    bt.logging.info(f"Pushed model to Hugging Face Hub at {repo_url}")

def get_url( synapse: pretrain.protocol.GetUrl ) -> pretrain.protocol.GetUrl:
    synapse.huggingface_url = repo_url
    return synapse

model_path = os.path.expanduser(f"~/pretrain-subnet/neurons/pretraining_model/{model_name}/model.bin")
timestamp = os.path.getmtime( model_path )
model = pretrain.model.get_model( )
update_model(model, model_path) 
model_weights = torch.load( model_path )
model.load_state_dict( model_weights )
bt.logging.success( f'Loaded model from: {model_path}' )

# === Axon ===
axon = bt.axon( 
    wallet = wallet, 
    config = config 
).attach( 
    forward_fn = get_url,
).start()
bt.logging.success( f'Started axon.' )

axon.start().serve( 
    subtensor = subtensor,
    netuid = pretrain.NETUID,
)
bt.logging.success( f'Served Axon.' )


# === Run ===
step = 1
subtensor.set_weights (
    netuid = pretrain.NETUID,
    wallet = wallet, 
    uids = [my_uid], 
    weights = [1.0], 
    wait_for_inclusion=False,
)
while True:
    try:
        bt.logging.success( f'Waiting for updated on {model_path}' )

        new_timestamp = os.path.getmtime( model_path )
        if new_timestamp != timestamp:
            bt.logging.info(f"Found newer model at {model_path}")
            model = pretrain.model.get_model()
            model_weights = torch.load(model_path)
            model.load_state_dict(model_weights)
            update_model(model, model_path)
            timestamp = new_timestamp

        time.sleep( 10 )
        step += 1

        if step % 100 == 0:
            subtensor.set_weights (
                netuid = pretrain.NETUID,
                wallet = wallet, 
                uids = [my_uid], 
                weights = [1.0], 
                wait_for_inclusion=False,
            )

    except Exception as e:
        bt.logging.error( traceback.format_exc() )
        continue




