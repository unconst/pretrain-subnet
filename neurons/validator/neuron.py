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
import time
import torch
import wandb
import typing
import asyncio
import traceback
import bittensor as bt
from transformers import AdamW

import pretrain
from background import background_loop
from foreground import foreground_loop

from forward import priority
from forward import blacklist
from forward import get_state

def init_wandb( self: object, type: str, uid: int, reinit=False ):
    """Starts a new wandb run."""
    if self.config.wandb.on:
        tags = [ type, f'uid:{uid}', self.wallet.hotkey.ss58_address, pretrain.__version__, str(pretrain.__spec_version__), f'netuid_{pretrain.NETUID}']
        return wandb.init(
            anonymous = "allow",
            reinit = reinit,
            project = self.config.wandb.project_name,
            entity = self.config.wandb.entity,
            config = self.config,
            mode = "offline" if self.config.wandb.offline else "online",
            dir = self.config.full_path,
            tags=tags,
        )
    else:
        return None

class Validator:

    def __init__( self, config ):        
        # === Logging ===
        bt.logging( config=config, logging_dir = config.full_path )
        bt.logging.info( config )

        # === Bittensor objects ===
        self.config = config
        self.wallet = bt.wallet(config=self.config)
        self.subtensor = bt.subtensor(config=self.config)
        self.metagraph = self.subtensor.metagraph(pretrain.NETUID)
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys: raise Exception("You are not registered.")
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        
        # === Init wandb ===
        self.wandb = init_wandb( self, type = 'validator', uid = self.uid )
    
        # === State ===
        self.best_average_loss = math.inf
        self.best_model_state = pretrain.model.get_model().to( self.config.device ).state_dict()
        self.global_state = {
            'n_successes': 0,
            'n_failures': 0,
            'n_exceptions': 0,
            'steps_per_second': 0.0,
            'last_query': time.time(),
            'query_time': 60,
            'uid_state': {},
            'n_steps': 0,
        }

        # === Axon Callbacks ===
        async def priority_fn( synapse: pretrain.protocol.GetState ) -> float: return await priority( self, synapse )
        async def blacklist_fn( synapse: pretrain.protocol.GetState ) -> typing.Tuple[bool, str]: return await blacklist( self, synapse )
        async def get_state( synapse: pretrain.protocol.GetState ) -> pretrain.protocol.GetState: return await get_state( self, synapse )

        # === Axon ===
        self.axon = bt.axon( 
            wallet = self.wallet, 
            config = self.config 
        ).attach( 
            forward_fn = get_state,
            priority_fn = priority_fn,
            blacklist_fn = blacklist_fn
        ).start()
        bt.logging.info(f"Served Axon {self.axon} on network: on network: {self.config.subtensor.chain_endpoint} with netuid: {pretrain.NETUID}")


    # === Validator entrypoint ===
    def run( self ):

        # === Start up axon===
        self.axon.start().serve( 
            subtensor = self.subtensor,
            netuid = pretrain.NETUID,
        )

        # === Main loop ===
        async def main_loop():
            bt.logging.success( 'Starting validator main loop.' )
            asyncio.create_task( background_loop( self ) )
            await asyncio.sleep( 5 )
            asyncio.run( foreground_loop( self ) )

        # === Start ===
        loop = asyncio.get_event_loop()
        try:
            def handle_exception(loop, context):
                if 'exception' in context:
                    bt.logging.error(f"Caught exception: {context['exception']}")
                else:
                    bt.logging.error(f"Caught exception, but no exception object was provided in the context. Context: {context}")
            # Run the main loop until completion.
            loop.set_exception_handler( handle_exception )
            loop.run_until_complete( main_loop() )

        # If we encounter an unexpected error, log it for debugging.
        except RuntimeError as e:
            bt.logging.error(e)
            traceback.print_exc()

        # If the user interrupts the program, gracefully exit.
        except KeyboardInterrupt:
            bt.logging.success("Keyboard interrupt detected. Exiting validator.")
            wandb.finish()
            exit()
