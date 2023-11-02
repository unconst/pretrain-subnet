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
import bittensor as bt

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
wallet = bt.wallet( config = config ) 
subtensor = bt.subtensor( config = config )
metagraph = subtensor.metagraph( pretrain.NETUID )
if wallet.hotkey.ss58_address not in metagraph.hotkeys: raise Exception("You are not registered. Use `btcli s recycle_register` to register.")

# === Init wandb ===
run_name = ''.join(random.choice( string.ascii_uppercase + string.digits ) for i in range(10))
config.uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
config.hotkey = wallet.hotkey.ss58_address
wand =  wandb.init(
    name = run_name,
    anonymous = "allow",
    reinit = False,
    project = 'openpretraining',
    entity = 'opentensor-dev',
    config = config,
)

model_path = os.path.expanduser( config.model_path )
timestamp = os.path.getmtime( model_path )
model = pretrain.model.get_model( config.model )
model_weights = torch.load( model_path )
model.load_state_dict( model_weights )

def get_run( synapse: pretrain.protocol.GetRun ) -> pretrain.protocol.GetRun:
    synapse.run_name = config.run_name
    return synapse

# === Axon ===
axon = bt.axon( 
    wallet = wallet, 
    config = config 
).attach( 
    forward_fn = get_run,
).start()

axon.start().serve( 
    subtensor = subtensor,
    netuid = pretrain.NETUID,
)

# === Run ===
while True:
    new_timestamp = os.path.getmtime( model_path )
    if new_timestamp != timestamp:
        model = pretrain.models.get_model( config.model )
        model_weights = torch.load( model_path )
        model.load_state_dict( model_weights )
        wandb.save("model_weights.pth")
    time.sleep( 1 )






