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
import wandb
import torch
import string
import random
import argparse
import pretrain
import traceback
import bittensor as bt

# === Config ===
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model_path", type=str, required=False, help="Override model path")
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
            "miner",
        )
    )
    if config.model_path == None:
        config.model_path = config.full_path + '/' + 'model.pth'
    if not os.path.exists( os.path.dirname(config.model_path) ):
        os.makedirs( os.path.dirname(config.model_path), exist_ok=True )
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

# === Init wandb ===
run_name = f'{my_uid}-' + ''.join(random.choice( string.ascii_uppercase + string.digits ) for i in range(10))
config.uid = my_uid
config.hotkey = wallet.hotkey.ss58_address
config.run_name = run_name
wandb_run = wandb.init(
    name = run_name,
    anonymous = "allow",
    reinit = False,
    project = 'openpretraining',
    entity = 'opentensor-dev',
    config = config,
    dir = config.full_path,
)

config.signature = wallet.hotkey.sign( wandb_run.id.encode() ).hex()
wandb.config.update( config )


timestamp = os.path.getmtime( config.model_path )
model = pretrain.model.get_model( )
model_weights = torch.load( config.model_path )
model.load_state_dict( model_weights )
bt.logging.success( f'Loaded model from: {config.model_path}' )

wandb.save( config.model_path )
bt.logging.success( f'Saved weights to wandb' )


wandb_run.finish()
