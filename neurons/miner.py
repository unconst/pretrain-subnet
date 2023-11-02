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
bt.logging( config = config )
bt.logging.success( config )
wallet = bt.wallet( config = config ) 
subtensor = bt.subtensor( config = config )
metagraph = subtensor.metagraph( pretrain.NETUID )
if wallet.hotkey.ss58_address not in metagraph.hotkeys: raise Exception("You are not registered. Use `btcli s recycle_register` to register.")
my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
bt.logging.success( f'You are registered with address: {wallet.hotkey.ss58_address} and uid: {my_uid}' )

# === Init wandb ===
run_name = f'u:{my_uid}-' + ''.join(random.choice( string.ascii_uppercase + string.digits ) for i in range(10))
config.uid = my_uid
config.hotkey = wallet.hotkey.ss58_address
config.run_name = run_name
wand =  wandb.init(
    name = run_name,
    anonymous = "allow",
    reinit = False,
    project = 'openpretraining',
    entity = 'opentensor-dev',
    config = config,
)
bt.logging.success( f'Started wandb run' )

model_path = os.path.expanduser( config.model_path )
timestamp = os.path.getmtime( model_path )
model = pretrain.model.get_model( )
model_weights = torch.load( model_path )
model.load_state_dict( model_weights )
bt.logging.success( f'Loaded model from: {model_path}' )

wandb.save( model_path )
bt.logging.success( f'Saved weights to wandb' )

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
    bt.logging.success( f'Waiting for updated on {model_path}' )

    new_timestamp = os.path.getmtime( model_path )
    if new_timestamp != timestamp:
        model = pretrain.model.get_model()
        model_weights = torch.load( model_path )
        model.load_state_dict( model_weights )
        wandb.save( model_path )
        timestamp = new_timestamp
        bt.logging.success( f'Found newer model at {model_path}' )

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

wandb.finish()





