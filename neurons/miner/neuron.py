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
import random
import asyncio
import pretrain
import traceback
import bittensor as bt
from helpers import init_wandb

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
        self.subtensor = bt.subtensor( config = self.config )
        self.metagraph = self.subtensor.metagraph( pretrain.NETUID )
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys: raise Exception("You are not registered. Use `btcli s recycle_register` to register.")
        self.uid = self.metagraph.hotkeys.index(self.wallet.hotkey.ss58_address)
        bt.logging.info(f"Wallet: {self.wallet}")
        bt.logging.info(f"Subtensor: {self.subtensor}")
        bt.logging.info(f"Metagraph: {self.metagraph}")
        bt.logging.info(f"Running miner on uid: {self.uid}")

        # === Init wandb ===
        self.wandb = init_wandb( self, type = 'miner', uid = self.uid )

        # === Init model state ===
        self.best_average_loss = math.inf
        self.best_model_state = pretrain.model.get_model().to( self.config.device ).state_dict()

        # === Init model ===
        self.model = pretrain.model.get_model().to( self.config.device )
        self.optimizer = torch.optim.AdamW( self.model.parameters(), lr = 0.001 )

        # === Axon Callbacks ===
        async def priority_fn( synapse: pretrain.protocol.GetState ) -> float: return await priority( self, synapse )
        async def blacklist_fn( synapse: pretrain.protocol.GetState ) -> typing.Tuple[bool, str]: return await blacklist( self, synapse )
        async def get_state_fn( synapse: pretrain.protocol.GetState ) -> pretrain.protocol.GetState: return await get_state( self, synapse )

        # === Axon ===
        self.axon = bt.axon( 
            wallet = self.wallet, 
            config = self.config 
        ).attach( 
            forward_fn = get_state_fn,
            priority_fn = priority_fn,
            blacklist_fn = blacklist_fn
        ).start()
        bt.logging.info(f"Served Axon {self.axon} on network: on network: {self.config.subtensor.chain_endpoint} with netuid: {pretrain.NETUID}")


    # === Train ===
    def train(self):
        # Get Random page. 
        random_pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages)]
        loader = pretrain.dataset.SubsetFalconLoader(
            batch_size = 3,
            sequence_length = 512,
            pages = random_pages
        )
        torch.backends.cudnn.benchmark = True

        # Run eval
        n_batches = 0
        average_loss = 0.0
        for i, batch in enumerate( loader ):
            inputs = batch.to( self.model.device )
            outputs = self.model( inputs, labels=inputs )
            outputs.loss.backward()
            average_loss += outputs.loss.detach().item()
            n_batches += 1
            bt.logging.success( f'Acc: step: {i} loss: {outputs.loss}' )
            self.optimizer.step()
            self.optimizer.zero_grad()
            torch.cuda.empty_cache()

        # Return average loss on batch.
        return average_loss / n_batches

    # === Miner entrypoint ===
    def run(self):

        # === Start up axon===
        self.axon.start().serve( 
            subtensor = self.subtensor,
            netuid = pretrain.NETUID,
        )

        # === Set active ping. ===
        self.subtensor.set_weights (
            netuid = pretrain.NETUID,
            wallet = self.wallet, 
            uids = [self.uid], 
            weights = [1.0], 
            wait_for_inclusion=False,
        )

        # === Global Loop ===
        bt.logging.info(f"Starting main loop")
        self.step = 0
        while True:
            try: 
                # Get Random page. 
                loss = self.train()
                if loss < self.best_average_loss:
                    self.best_average_loss = loss
                    self.best_model_state = self.model.state_dict()
                    bt.logging.success(f"New best average loss: {self.best_average_loss}")

                # Resync the metagraph.
                self.metagraph = self.subtensor.metagraph( pretrain.NETUID )
                self.step += 1

                # Set ping weights every 50 blocks.
                if self.step % 5 == 0:
                    self.subtensor.set_weights (
                        netuid = pretrain.NETUID,
                        wallet = self.wallet, 
                        uids = [self.uid], 
                        weights = [1.0], 
                        wait_for_inclusion=False,
                    )

            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                break

            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue