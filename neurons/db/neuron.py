import time
import helpers
import traceback
import typing
import pretrain
import bittensor as bt

from helpers import init_wandb
from forward import priority
from forward import blacklist
from forward import compute_gradients

class DB:

    def __init__( self, config ):

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

        # === Axon Callbacks ===
        async def priority_fn( synapse: pretrain.protocol.GetState ) -> float: return await priority( self, synapse )
        async def blacklist_fn( synapse: pretrain.protocol.GetState ) -> typing.Tuple[bool, str]: return await blacklist( self, synapse )
        async def get_state( synapse: pretrain.protocol.GetState ) -> pretrain.protocol.GetState: return await get_state( self, synapse )

        async def priority_fn( synapse: pretrain.protocol.ApplyGrads ) -> float: return await priority( self, synapse )
        async def blacklist_fn( synapse: pretrain.protocol.ApplyGrads ) -> typing.Tuple[bool, str]: return await blacklist( self, synapse )
        async def apply_grads( synapse: pretrain.protocol.ApplyGrads ) -> pretrain.protocol.ApplyGrads: return await apply_grads( self, synapse )

        # === Axon ===
        self.axon = bt.axon( 
            wallet = self.wallet, 
            config = self.config 
        ).attach( 
            forward_fn = get_state,
            priority_fn = priority_fn,
            blacklist_fn = blacklist_fn
        ).attach( 
            forward_fn = get_state,
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
                self.block += 1

            # If someone intentionally stops the miner, it'll safely terminate operations.
            except KeyboardInterrupt:
                self.axon.stop()
                bt.logging.success("Miner killed by keyboard interrupt.")
                break

            # In case of unforeseen errors, the miner will log the error and continue operations.
            except Exception as e:
                bt.logging.error(traceback.format_exc())
                continue