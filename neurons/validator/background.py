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

import math
import torch
import asyncio
import pretrain
import bittensor as bt

from rich import print
from rich.columns import Columns
from rich.panel import Panel
from rich.text import Text

def normalize(value, min_value, max_value):
    return (value - min_value) / (max_value - min_value + 0.00000001)

def get_weight_color(value, min_value, max_value):
    norm_value = normalize(value, min_value, max_value)
    return f'\033[38;2;0;255;0m'  # RGB color code for varying shades of green

def pretty_print_weights(self):
    mean_weight = sum(self.weights)/len(self.weights)
    items = [
        Text(f"UID: {index}, Weight: {weight}", style=f"color: {'green' if weight > mean_weight else 'red' if weight <= mean_weight else 'yellow'}" )
        for index, weight in enumerate(self.weights.tolist())
    ]
    columns = Columns(items, equal=True, expand=True)
    panel = Panel(columns, title="Weights")
    print(panel)

# Returns the scores for current miners.
def compute_weights( self: object ):
    # Fill weights. weight_i = exp( -score_i ) / SUM_j exp( -score_j )
    # Where score_i is the moving average of negative of the MSE between grads returned and grads computed.
    self.weights = torch.zeros_like( self.metagraph.S )
    for uid in self.metagraph.uids.tolist():
        self.weights[ uid ] = math.exp( self.global_state['uid_scores'][ uid ] ) if uid in self.global_state['uid_scores'] else 0.0
    # Normalize the scores to 1.0
    self.weights = self.weights / self.weights.sum()
    return self.weights

def set_weights( self: object ):
    bt.logging.info(f"Setting weights: {self.weights}")
    # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
    # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
    result = self.subtensor.set_weights(
        netuid = pretrain.NETUID,  # Subnet to set weights on.
        wallet = self.wallet,  # Wallet to sign set weights using hotkey.
        uids = self.metagraph.uids,  # Uids of the miners to set weights for.
        weights = self.weights,  # Weights to set for the miners.
        wait_for_inclusion=False,
    )
    if result:
        bt.logging.success("Successfully set weights.")
    else:
        bt.logging.error("Failed to set weights.")

async def background_loop( self: object ):

    # === Getting availble ===
    bt.logging.success( 'Starting validator background loop.' )
    self.background_step = 0
    while True:

        try:
            # Wait for one block.
            bt.logging.debug("Background ideling...")
            await asyncio.sleep( 60 ) 
            self.background_step += 1

            # Resync the metagraph.
            self.metagraph = self.subtensor.metagraph( pretrain.NETUID )
            self.weights = compute_weights( self )
            pretty_print_weights( self )

            # Set weights every 50 blocks.    
            if self.background_step % 50 == 0:
                bt.logging.success('Setting weights on chain.')
                set_weights( self )
    
        except Exception as e:
            bt.logging.error( f"Caught exception in background loop with error: {e}" )
          