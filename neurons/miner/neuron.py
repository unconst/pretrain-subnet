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
import asyncio
import pretrain
import traceback
import bittensor as bt
from helpers import init_wandb

from forward import priority_fn
from forward import blacklist_fn
from forward import compute_gradients

class Miner:

    # === Init ===
    def __init__(self, config):

        # === Config ===
        self.config = config

        # === Logging ===
        bt.logging(config=config, logging_dir=config.full_path)
        bt.logging.info( f"Running miner for subnet: { pretrain.NETUID } on network: {config.subtensor.chain_endpoint} with config:")
        bt.logging.info(self.config)

        # === Bittensor objects ===
        self.wallet = bt.wallet( config = self.config ) 
        self.subtensor = bt.subtensor( config = self.config)
        self.metagraph = self.subtensor.metagraph( pretrain.NETUID )
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys: raise Exception("You are not registered. Use `btcli s recycle_register` to register.")
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Wallet: {self.wallet}")
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(f"Metagraph: {self.metagraph}")
        bt.logging.info(f"Running miner on uid: {self.uid}")

        # === Init wandb ===
        self.wandb = init_wandb( self, type = 'miner', uid = self.uid )

        # === Locks ===
        # Limits GPU usage to 1 request at a time for space considerations. In practice, we would
        # shuttle multiple requests across multiple machines.
        self.gpu_lock = asyncio.Lock() 
        # Limits the number of queries that can pass the header checks in the blacklist.
        # Increasing this number allow for the miner to download more requests concurrently.
        self.global_forward_lock = asyncio.Semaphore( self.config.max_concurrent_forward_requests ) 

        # === Axon Callbacks ===
        async def blacklist_fn( synapse: pretrain.protocol.ComputeGradients ) -> float: return blacklist_fn( self, synapse )
        async def priority_fn( synapse: pretrain.protocol.ComputeGradients ) -> float: return priority_fn( self, synapse )
        async def compute_gradients( synapse: pretrain.protocol.ComputeGradients ) -> float: return compute_gradients( self, synapse )

        # === Axon ===
        self.axon = bt.axon( 
            wallet = self.wallet, 
            config = self.config 
        ).attach( 
            forward_fn = compute_gradients,
            priority_fn = priority_fn,
            blacklist_fn = blacklist_fn
        ).start()
        bt.logging.info(f"Served Axon {self.axon} on network: on network: {self.config.subtensor.chain_endpoint} with netuid: {pretrain.NETUID}")

    # === Miner entrypoint ===
    def run(self):

        # === Start up axon===
        self.axon.start().serve( 
            subtensor = self.subtensor,
            netuid = pretrain.NETUID,
        )

        # === Global Loop ===
        bt.logging.info(f"Starting main loop")
        self.block = 0
        while True:
            try: 

                # Resync the metagraph.
                self.metagraph = self.subtensor.metagraph( pretrain.NETUID )
                self.block = self.metagraph.block.item()
                time.sleep( bt.__blocktime__ )
                block += 1

            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                break

            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue