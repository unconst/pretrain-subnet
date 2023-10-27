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
import time
import torch
import time
import typing
import asyncio
import argparse
import traceback
import bittensor as bt

# import this repo
import pretrain
import helpers

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( '--device', type = str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the miner on.' )
    parser.add_argument( "--max_concurrent_forward_requests", type=int, help="Maximum number of concurrent forward requests.", default=1 )
    parser.add_argument( "--wandb.off", action="store_true", help="Turn off wandb.", default=False)
    parser.add_argument( "--wandb.project_name", type=str, help="The name of the project where you are sending the new run.", default="openpretraining" )
    parser.add_argument( "--wandb.entity", type=str, help="An entity is a username or team name where youre sending runs.", default="opentensor-dev" )
    parser.add_argument( "--wandb.offline", action="store_true", help="Runs wandb in offline mode.", default=False,)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            pretrain.NETUID,
            "miner",
        )
    )
    # Ensure the directory for logging exists, else create one.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)
    return config

# Main takes the config and starts the miner.
def main(config):

    # === Logging ===
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info( f"Running miner for subnet: { pretrain.NETUID } on network: {config.subtensor.chain_endpoint} with config:")
    bt.logging.info(config)

    # === Bittensor objects ===
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph( pretrain.NETUID )
    if wallet.hotkey.ss58_address not in metagraph.hotkeys: raise Exception("You are not registered.")
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    helpers.init_wandb( wallet = wallet, config = config, type = 'validator', uid = my_subnet_uid )
    bt.logging.info(f"Wallet: {wallet}")
    bt.logging.info(f"Subtensor: {subtensor}")
    bt.logging.info(f"Metagraph: {metagraph}")
    bt.logging.info(f"Running miner on uid: {my_subnet_uid}")

    # === Locks ===
    # Limits GPU usage to 1 request at a time for space considerations. In practice, we would
    # shuttle multiple requests across multiple machines.
    gpu_lock = asyncio.Lock() 
    # Limits the number of queries that can pass the header checks in the blacklist.
    # Increasing this number allow for the miner to download more requests concurrently.
    global_forward_lock = asyncio.Semaphore( config.max_concurrent_forward_requests ) 

    # === Blacklist ===
    async def blacklist_fn( synapse: pretrain.protocol.ComputeGradients ) -> typing.Tuple[bool, str]:
        # Locks requests to only allowing max_concurrent_forward_requests at a time.
        # After the blacklist the full synapse is pulled into memory so we want to limit
        # the number here.
        async with global_forward_lock:
            # Check if the hotkey is in the metagraph.
            if synapse.dendrite.hotkey not in metagraph.hotkeys:
                    # Allow query through.
                    return True, "Unrecognized hotkey"
            # Blacklist query.
            return False, "Hotkey recognized!"

    # === Priority ===
    async def priority_fn( synapse: pretrain.protocol.ComputeGradients ) -> float:
        # Priority is stake based.
        caller_uid = metagraph.hotkeys.index( synapse.dendrite.hotkey )  
        prirority = float(metagraph.S[caller_uid]) 
        return prirority

    # === Forward ===
    async def compute_gradients( synapse: pretrain.protocol.ComputeGradients ) -> pretrain.protocol.ComputeGradients:
        """
        Compute the gradients for a given model based on passed batch, sequence length and pages.

        Args:
            synapse (pretrain.protocol.ComputeGradients): Object containing serialized model state 
                                                        and timeout value.

        Returns:
            pretrain.protocol.ComputeGradients: Object containing the serialized gradients.
        """
        wandb_event = {}
        wandb_event['init_forward'] = time.time()
        bt.logging.success(f'Received request for synapse: {synapse.axon.hotkey}')
        # Lock the model since concurrent accumulation to the model will poision the gradients we 
        # are computing. In practice we would shuttle multiple requests across multiple machines.
        # This lock is not necessary if we are only running a single cuda core.
        async with gpu_lock:
            wandb_event['forward_start'] = time.time()
            # Move the model to the same device as the synapse
            local_model = pretrain.model.get_model()
            local_model.load_state_dict( synapse.deserialize() )
            local_model = local_model.to( config.device )
            bt.logging.success( f'Aquired GPU space for query.' )
            # Compute gradients on the model.
            grads_dict = helpers.compute_gradients_on_model(
                model = local_model,
                batch_size = synapse.batch_size,
                sequence_length = synapse.sequence_length,
                pages = synapse.pages
            )
        # Serialize accumulated gradients into the synapse object
        synapse.serialize( state_dict = grads_dict )
        bt.logging.success( f'Serialized response gradients.' )
        wandb_event['forward_end'] = time.time()
        wandb_event['forward_full_time'] = time.time() - wandb_event['init_forward']
        wandb_event['forward_process_time'] = time.time() - wandb_event['forward_start']
        return synapse

    # === Axon ===
    axon = bt.axon( 
        wallet = wallet, 
        config = config 
    ).attach( 
        forward_fn = compute_gradients,
        priority_fn = priority_fn,
        blacklist_fn = blacklist_fn
    ).start()
    bt.logging.info(f"Served Axon {axon} on network: on network: {config.subtensor.chain_endpoint} with netuid: {pretrain.NETUID}")

    # === Global Loop ===
    bt.logging.info(f"Starting main loop")
    block = 0
    while True:
        try: 
            time.sleep( bt.__blocktime__ )
            block += 1

        # If someone intentionally stops the miner, it'll safely terminate operations.
        except KeyboardInterrupt:
            axon.stop()
            bt.logging.success("Miner killed by keyboard interrupt.")
            break

        # In case of unforeseen errors, the miner will log the error and continue operations.
        except Exception as e:
            bt.logging.error(traceback.format_exc())
            continue


# This is the main function, which runs the miner.
if __name__ == "__main__":
    main(get_config())
