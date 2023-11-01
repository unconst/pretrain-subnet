
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

import time
import typing
import pretrain
from rich import print
from rich.table import Table
import bittensor as bt

from helpers import compute_gradients_on_model

# === Blacklist ===
async def blacklist( self, synapse: pretrain.protocol.ComputeGradients ) -> typing.Tuple[bool, str]:
    # Locks requests to only allowing max_concurrent_forward_requests at a time.
    # After the blacklist the full synapse is pulled into memory so we want to limit
    # the number here.
    async with self.global_forward_lock:
        # Check if the hotkey is in the metagraph.
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                # Allow query through.
                return True, "Unrecognized hotkey"
        # Blacklist query.
        return False, "Hotkey recognized!"

# === Priority ===
async def priority( self, synapse: pretrain.protocol.ComputeGradients ) -> float:
    # Priority is stake based.
    caller_uid = self.metagraph.hotkeys.index( synapse.dendrite.hotkey )  
    prirority = float(self.metagraph.S[caller_uid]) 
    return prirority

# === Forward Get State ===
async def get_state( self, synapse: pretrain.protocol.GetState ) -> pretrain.protocol.GetState:
    synapse.serialize( state_dict = self.model.state_dict() )
    return synapse

# === Forward Apply Grads ===
async def apply_grads( self, synapse: pretrain.protocol.ApplyGrads ) -> pretrain.protocol.ApplyGrads:

    grads_dict = synapse.deserialize()
    self.model.zero_grad()
    for name_j, param_j in self.model.named_parameters():
        if name_j in grads_dict and grads_dict[name_j] is not None:
            param_j.grad = param_j.grad + grads_dict[name_j].to('cpu') if param_j.grad is not None else grads_dict[name_j].to('cpu')

    self.optimizer.step()
    self.model.zero_grad()

    # Empty state and respond.
    synapse.serialize( state_dict = {} )
    return synapse