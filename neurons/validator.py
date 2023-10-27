# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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

# Step 1: Import necessary libraries and modules
import os
import time
import torch
import typing
import random
import asyncio
import argparse
import traceback
import bittensor as bt
from transformers import AdamW

# import this repo
import pretrain
import helpers

# Step 2: Set up the configuration parser
# This function is responsible for setting up and parsing command-line arguments.
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--alpha", default=0.9, type=float, help="The weight moving average scoring." )
    parser.add_argument( '--learning_rate', default=1e-4, type=float, help='Learning rate for the optimizer.' )
    parser.add_argument( '--batch_size', type=int, default=3, help='Eval batch size' )
    parser.add_argument( '--sequence_length', type=int, default=512, help='Eval sequence length' )
    parser.add_argument( '--n_eval_steps', default=10, type=int, help='Number of eval steps.' )
    parser.add_argument( '--device', type = str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the miner on.' )
    # Adds subtensor specific arguments i.e. --subtensor.chain_endpoint ... --subtensor.network ...
    bt.subtensor.add_args(parser)
    # Adds logging specific arguments i.e. --logging.debug ..., --logging.trace .. or --logging.logging_dir ...
    bt.logging.add_args(parser)
    # Adds wallet specific arguments i.e. --wallet.name ..., --wallet.hotkey ./. or --wallet.path ...
    bt.wallet.add_args(parser)
    # Parse the config (will take command-line arguments if provided)
    # To print help message, run python3 template/validator.py --help
    config = bt.config(parser)
    # Step 3: Set up logging directory
    # Logging is crucial for monitoring and debugging purposes.
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            pretrain.NETUID,
            "validator",
        )
    )
    # Ensure the logging directory exists.
    if not os.path.exists(config.full_path):
        os.makedirs(config.full_path, exist_ok=True)

    # Return the parsed config.
    return config


def main(config):
    # Set up logging with the provided configuration and directory.
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(
        f"Running validator for subnet: {pretrain.NETUID} on network: {config.subtensor.chain_endpoint} with config:"
    )
    # Log the configuration for reference.
    bt.logging.info(config)

    # Step 4: Build Bittensor validator objects
    # These are core Bittensor classes to interact with the network.
    bt.logging.info("Setting up bittensor objects.")

    # The wallet holds the cryptographic key pairs for the validator.
    wallet = bt.wallet(config=config)
    bt.logging.info(f"Wallet: {wallet}")

    # The subtensor is our connection to the Bittensor blockchain.
    subtensor = bt.subtensor(config=config)
    bt.logging.info(f"Subtensor: {subtensor}")

    # Dendrite is the RPC client; it lets us send messages to other nodes (axons) in the network.
    dendrite = bt.dendrite(wallet=wallet)
    bt.logging.info(f"Dendrite: {dendrite}")

    # The metagraph holds the state of the network, letting us know about other validators and miners.
    metagraph = subtensor.metagraph(pretrain.NETUID)
    bt.logging.info(f"Metagraph: {metagraph}")

    # Step 5: Connect the validator to the network
    if wallet.hotkey.ss58_address not in metagraph.hotkeys:
        bt.logging.error(
            f"\nYour validator: {wallet} if not registered to chain connection: {subtensor} \nRun btcli register and try again."
        )
        exit()

    # Each validator gets a unique identity (UID) in the network for differentiation.
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    bt.logging.info(f"Running validator on uid: {my_subnet_uid}")

    # Step 6: Set up initial scoring weights for validation
    bt.logging.info("Building validation weights.")
    scores = {}
    bt.logging.info(f"Weights: {scores}")

    # Step 7: Build mode for validation.
    model = pretrain.model.get_model().to( config.device )
    
    # Step 8: Build optimizer for validation.
    optimizer = AdamW( model.parameters(), lr=config.learning_rate )
    
    # Build forward locks.
    gpu_lock = asyncio.Lock()
    model_lock = asyncio.Lock()
    max_concurrent_forward = asyncio.Semaphore( 1 )

    # Define forward function.
    async def forward( ):

        # Acquire the forward lock to limit concurrent forward calls.
        async with max_concurrent_forward:
            bt.logging.success( 'Started validator forward task.' )

            # Get a random miner axon.
            available_uids = [uid.item() for uid in metagraph.uids if metagraph.axons[uid].is_serving]
            if len(available_uids) == 0: return
            uid = random.choice(available_uids)
            bt.logging.success( f'Selected uid:{uid}' )


            # Build the query for miner
            synapse = pretrain.protocol.ComputeGradients( 
                batch_size = 8,
                sequence_length = 512,
                pages = [ 0 ]
            )
            # Get current model state.
            async with model_lock:
                query_model_state = model.state_dict()

            # Serialize the model state into the synapse object.
            synapse.serialize( state_dict = query_model_state ) 
            bt.logging.success( f'Serialized state.' )

            # Make the forward query.
            dendrite = bt.dendrite( wallet = wallet )
            bt.logging.success( f'Sent request.' )
            response = await dendrite.forward( metagraph.axons[ uid ], synapse, timeout = 60, deserialize = False )
            await dendrite.close_session()

            # Check for failures.
            if not response.is_success: 
                # Failed requests get a mse increase of 1 added.
                scores[ uid ] = scores[uid] + 1 if uid in scores else 1
                return
            bt.logging.success( f'Response success.' )

            # Deserialize the gradients from the response.
            grads_dict = response.deserialize()

            # Score the local model by comparing the gradients generated locally vs 
            # the gradients generated by the miner.
            async with gpu_lock:
                bt.logging.success( f'Aquired GPU space.' )
                # Load the models weights into a new model and position it on the gpu.
                eval_model = pretrain.model.get_model()
                eval_model.load_state_dict( query_model_state )
                eval_model.to( config.device )
                bt.logging.success( f'Created eval model on GPU')
                # Compute the gradients on the model.
                local = helpers.compute_gradients_on_model(
                    model = eval_model,
                    batch_size = 8,
                    sequence_length = 512,
                    pages = [ 0 ]
                )
                bt.logging.success( f'Finished local gradient computation' )
                # Accumulate the MSE of gradients difs.
                mse = helpers.mse_gradients( local, grads_dict )
                bt.logging.success( f'Computed MSE: {mse}' )
                scores[ uid ] = scores[uid] + mse if uid in scores else mse
                bt.logging.success( f'Updated score: {scores[uid]}' )

            # Apply the gradients to the local model.
            async with model_lock:
                bt.logging.success( f'Aquired model lock.' )
                # Accumulate grads on local model.
                for name_j, param_j in model.named_parameters():
                    if name_j in grads_dict and grads_dict[name_j] is not None:
                        param_j.grad = param_j.grad + grads_dict[name_j] if param_j.grad is not None else grads_dict[name_j]
                bt.logging.success( f'Applied gradients to model.' )

                # Take a step using the optimizer to update model parameters
                optimizer.step()
                bt.logging.success( f'Applied step.' )

                # Zero out the gradients to free up memory and avoid accidental accumulation in the next iteration
                model.zero_grad()

            # Finish.
            bt.logging.success( f'Finished forward.' )


    # Training loop.
    async def training_loop():
        bt.logging.success( 'Starting validator training loop.' )
        while True:
            # Create a task for each request and add it to the event loop
            asyncio.create_task( forward() )
            # Wait for a short time before creating the next task
            await asyncio.sleep( 0.1 )

    # Background loop.
    async def background_loop():
        bt.logging.success( 'Starting validator background loop.' )
        global metagraph
        block = 0
        while True:
            # Wait for one block.
            bt.logging.success("Background ideling...")
            await asyncio.sleep( bt.__blocktime__ )
            metagraph = subtensor.metagraph( pretrain.NETUID)
            bt.logging.success("End Background step.")
            block += 1               

    # A function that runs a continuous loop of requests with a semaphore
    async def main_loop():
        bt.logging.success( 'Starting validator main loop.' )
        asyncio.run( training_loop() )
        asyncio.run( background_loop() )

    # Run the main loop until completion.
    loop = asyncio.get_event_loop()
    try:
        # Run the main loop until completion.
        loop.run_until_complete(main_loop())

    # If we encounter an unexpected error, log it for debugging.
    except RuntimeError as e:
        bt.logging.error(e)
        traceback.print_exc()

    # If the user interrupts the program, gracefully exit.
    except KeyboardInterrupt:
        bt.logging.success("Keyboard interrupt detected. Exiting validator.")
        exit()

# The main function parses the configuration and runs the validator.
if __name__ == "__main__":
    # Parse the configuration.
    config = get_config()
    # Run the main function.
    main(config)
