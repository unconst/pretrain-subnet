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
import torch
import argparse
import bittensor as bt
from transformers import AdamW

import pretrain

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--device', type = str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the miner on.' )
    parser.add_argument( "--sync", action="store_true", help="Turn on sync training", default=False)
    parser.add_argument( "--wandb.on", action="store_true", help="Turn on wandb.", default=False)
    parser.add_argument( "--wandb.project_name", type=str, help="The name of the project where you are sending the new run.", default="openpretraining" )
    parser.add_argument( "--wandb.entity", type=str, help="An entity is a username or team name where youre sending runs.", default="opentensor-dev" )
    parser.add_argument( "--wandb.offline", action="store_true", help="Runs wandb in offline mode.", default=False,)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
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

if __name__ == "__main__":
    neuron.DB( get_config() ).run()
