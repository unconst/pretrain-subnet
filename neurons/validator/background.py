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

def pretty_print_weights(self):
    items = [Text(f"Index: {index}, Weight: {weight}") for index, weight in enumerate( self.weights.items() )]
    columns = Columns(items, equal=True, expand=True)
    panel = Panel(columns, title="Weights")
    print(panel)

# Returns the scores for current miners.
def compute_weights( self: object ):
    # Fill weights. weight_i = exp( -score_i ) / SUM_j exp( -score_j )
    # Where score_i is the moving average of negative of the MSE between grads returned and grads computed.
    self.weights = torch.zeros_like( self.metagraph.S )
    for uid in self.metagraph.uids:
        self.weights[ uid ] = math.exp( self.scores[ uid ] ) if uid in self.scores else 0.0
    # Normalize the scores to 1.0
    self.weights = torch.nn.functional.normalize( self.weights, p=1.0, dim=0)
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
        wait_for_inclusion=True,
    )
    if result:
        bt.logging.success("Successfully set weights.")
    else:
        bt.logging.error("Failed to set weights.")

async def background_loop( self: object ):

    bt.logging.success( 'Starting validator background loop.' )
    self.block = 0
    while True:

        try:
            # Wait for one block.
            bt.logging.debug("Background ideling...")
            await asyncio.sleep( 1 )

            # Resync the metagraph.
            self.weights = compute_weights( self )
            self.metagraph = self.subtensor.metagraph( pretrain.NETUID )
            self.block = self.metagraph.block.item()
            self.weights = compute_weights( self )
            pretty_print_weights( self )

            # Log weights.
            bt.logging.debug(f"Weights: {self.weights}")

            # Set weights every 50 blocks.    
            if self.block % 50 == 0:
                set_weights( self )
    
        except Exception as e:
            bt.logging.error( f"Caught exception in background loop with error: {e}" )
          