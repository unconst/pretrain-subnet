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
import wandb
import asyncio
import traceback
import bittensor as bt
from transformers import AdamW

# import this repo
import pretrain
from helpers import init_wandb
from background import background_loop
from foreground import foreground_loop

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
        self.available_uids = self.metagraph.uids.tolist()
        
        # === Init wandb ===
        self.wandb = init_wandb( self, type = 'validator', uid = self.uid )

        # === Training objects ===
        self.scores = {}
        self.model = pretrain.model.get_model().to( self.config.device )
        self.optimizer = AdamW( self.model.parameters(), lr = self.config.learning_rate )
    
        # === Locks ===
        self.gpu_lock = asyncio.Lock()
        self.model_lock = asyncio.Lock()
        self.forward_lock = asyncio.Semaphore( self.config.max_concurrent_forward )
        self.per_uid_locks = { i: asyncio.Semaphore( self.config.max_concurrent_forward_per_uid ) for i in range(256) }

        # === State ===
        self.global_state = {
            'n_successes': 0,
            'n_failures': 0,
            'n_exceptions': 0,
            'n_pages': 0.0,
            'n_steps': 0,
            'steps_per_second': 0.0,
            'last_query': time.time(),
        }

    # === Validator entrypoint ===
    def run( self ):

        # === Main loop ===
        async def main_loop():
            bt.logging.success( 'Starting validator main loop.' )
            asyncio.create_task( background_loop( self ) )
            asyncio.run( foreground_loop( self ) )

        # === Start ===
        loop = asyncio.get_event_loop()
        try:
            def handle_exception( loop, context ):
                bt.logging.error(f"Caught exception: {context['exception']}")
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
