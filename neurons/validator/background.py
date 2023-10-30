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
    try:
        green_intensity = int(255 * norm_value)
    except:
        green_intensity = 1
    return f'\033[38;2;0;{green_intensity};0m'  # RGB color code for varying shades of green

def pretty_print_weights(self):
    min_weight = min(self.weights)
    max_weight = max(self.weights)
    items = [
        Text(f"Index: {index}, Weight: {weight}", style=get_weight_color(weight, min_weight, max_weight))
        for index, weight in enumerate(self.weights.tolist())
    ]
    columns = Columns(items, equal=True, expand=True)
    panel = Panel(columns, title="Weights")
    print(panel)

def get_available_uids( self: object ):
    available_uids = []
    dendrite = bt.dendrite( wallet = self.wallet )
    serving_uids = [uid.item() for uid in self.metagraph.uids if self.metagraph.axons[uid].is_serving]
    serving_axons = [ self.metagraph.axons[uid] for uid in serving_uids ]
    ping_responses = dendrite.query( serving_axons )
    for resp, uid in list(zip( ping_responses, serving_uids) ):
        if resp.is_success:
            available_uids.append( uid )
    return available_uids

# Returns the scores for current miners.
def compute_weights( self: object ):
    # Fill weights. weight_i = exp( -score_i ) / SUM_j exp( -score_j )
    # Where score_i is the moving average of negative of the MSE between grads returned and grads computed.
    self.weights = torch.zeros_like( self.metagraph.S )
    for uid in self.metagraph.uids.tolist():
        if uid == 230:
            print (uid, math.exp( self.scores[ uid ] ) )
        self.weights[ uid ] = math.exp( self.scores[ uid ] ) if uid in self.scores else 0.0
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
            await asyncio.sleep( bt.__blocktime__ )

            # Resync the metagraph.
            #self.available_uids = get_available_uids( self )
            self.metagraph = self.subtensor.metagraph( pretrain.NETUID )
            self.block = self.metagraph.block.item()
            self.weights = compute_weights( self )
            pretty_print_weights( self )
            bt.logging.success(f"Available: {self.available_uids}")

            # Set weights every 50 blocks.    
            if self.block % 50 == 0:
                set_weights( self )
    
        except Exception as e:
            bt.logging.error( f"Caught exception in background loop with error: {e}" )
          