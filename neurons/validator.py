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
    async_model_lock = asyncio.Lock()
    model = pretrain.model.get_model().to( config.device )
    model.train()
    
    # Step 8: Build optimizer for validation.
    optimizer = AdamW( model.parameters(), lr=config.learning_rate )

    async def apply_grads_to_model_and_step(grads: typing.List[typing.Dict[str, torch.Tensor]]):
        """
        Applies gradients received from multiple workers to the model and takes an optimizer step.

        Args:
            grads (typing.List[typing.Dict[str, torch.Tensor]]): List of dictionaries. Each dictionary 
                represents gradients from a worker where keys are parameter names and values are 
                corresponding gradients.
        """
        # Lock this function to ensure that only one worker can apply gradients at a time
        async with async_model_lock:

            # Zero out any previous gradients of the model to ensure clean accumulation
            model.zero_grad()

            # Move the model to the device specified in the config (usually GPU)
            model.to(config.device)

            # If grads is not a list, convert it into a list for uniform processing
            if not isinstance(grads, list):
                grads = [grads]

            # Accumulate gradients from all workers
            for grad_dict in grads:
                for name_j, param_j in model.named_parameters():
                    if name_j in grad_dict:
                        if grad_dict[name_j] is not None:
                            grad_ij = grad_dict[name_j]
                            # Check if the model parameter has not been initialized with gradients yet
                            if param_j.grad is None:
                                param_j.grad = grad_ij.to(config.device)
                            else:
                                param_j.grad += grad_ij.to(config.device)
                        else:
                            bt.logging.trace(f'remote_grads[{name_j}] is None')
                    else:
                        bt.logging.trace(f'{name_j} not in remote_grads:')

            # Average the accumulated gradients over the number of workers
            for _, param_j in model.named_parameters():
                if param_j.grad is not None:
                    param_j.grad /= len(grads)

            # Take a step using the optimizer to update model parameters
            optimizer.step()

            # Zero out the gradients to free up memory and avoid accidental accumulation in the next iteration
            model.zero_grad()

            # Clear CUDA cache to free up GPU memory. This can help in reducing memory fragmentation
            torch.cuda.empty_cache()

    # Function for computing loss on a subset of the dataset.
    dataloader = pretrain.dataset.get_dataloader( config.batch_size, config.sequence_length )
    async def compute_current_loss_on_subset(n_steps: int) -> float:
        """
        Computes the average loss of the model on a subset of data.

        Args:
            n_steps (int): Number of steps (batches) over which the loss should be computed.

        Returns:
            float: The average loss over the specified number of steps.
        """
        # Lock this function to ensure that only one worker can apply gradients at a time
        async with async_model_lock:
        
            # Initialize counters for steps and the cumulative loss
            step = 0
            total_loss = 0

            # Move the model to the appropriate device (e.g., GPU) for computation
            model.to(config.device)

            # Since this function is only for evaluation (i.e., we only need the forward pass),
            # we use 'torch.no_grad()' to inform PyTorch that it doesn't need to track 
            # or compute gradients, which saves memory and speeds up the process.
            with torch.no_grad():
                while True:
                    # Fetch the next batch of data from the dataloader
                    batch = next(dataloader)

                    # Move the batch data to the same device as the model
                    inputs = batch.to(config.device)
                    
                    # Compute the model's predictions for the given inputs
                    outputs = model(inputs, labels=inputs)

                    # Accumulate the loss value for this batch to the total loss
                    total_loss += outputs.loss.detach().item()
                    
                    # Log the current step's loss for monitoring
                    bt.logging.info(f'Eval Step: {step}, Loss: {outputs.loss.item()}')

                    # Check if we've reached the required number of steps; if so, break out of the loop
                    if step >= n_steps:
                        break
                    else:
                        step += 1

                # Delete the large tensor variables to free up memory. 
                # This is especially helpful when working with limited GPU memory.
                del batch
                del inputs
                del outputs

                # Explicitly empty the GPU cache to further ensure memory is released
                torch.cuda.empty_cache()

            # Return the average loss value over the number of steps
            return total_loss / n_steps

    def get_random_available_miner_axon() -> typing.Optional[int]:
        """
        Fetches a random UID (User ID) of a miner axon that is currently serving.

        Returns:
            typing.Optional[int]: UID of a randomly selected available miner axon. 
                                Returns None if no miner axons are available.
        """
        # Fetch the UIDs of all miner axons that are currently serving.
        # List comprehension iterates through all UIDs and checks if the corresponding axon is serving.
        available_uids = [uid.item() for uid in metagraph.uids if metagraph.axons[uid].is_serving]

        # Check if there are no available miner axons.
        if len(available_uids) == 0: 
            return None

        # Randomly select a UID from the available UIDs.
        random_miner_uid = random.choice(available_uids)

        return random_miner_uid
    
    # Function for forwarding to a random miner.
    max_concurrent_forwards_semaphore = asyncio.Semaphore( 10 )
    async def forward( ):
        bt.logging.success( 'Starting validator forward.' )

        # Get a random miner axon.
        uid = get_random_available_miner_axon(  )

        async with max_concurrent_forwards_semaphore:
            # Get axon of miner.
            axon = metagraph.axons[ uid ]

            # Build the query for miner
            synapse = pretrain.protocol.ComputeGradients()
            synapse.serialize( state_dict = model.state_dict() ) 

            # Make the broadcast query
            dendrite = bt.dendrite( wallet = wallet )
            grads = await dendrite.forward( axon, synapse, timeout = 60, deserialize = True )
            await dendrite.close_session()

            # Apply grads to model and step.
            apply_grads_to_model_and_step( grads )

            # Compute current loss on subset of dataset.
            loss = compute_current_loss_on_subset( n_steps = config.n_eval_steps )
            return loss
        
    # Training loop
    async def training_loop():
        bt.logging.success( 'Starting validator training loop.' )
        # Create a semaphore with a maximum of 10 concurrent tasks
        while True:
            # Create a task for each request and add it to the event loop
            asyncio.create_task( forward() )
            # Wait for a short time before creating the next task
            await asyncio.sleep( 0.1 )

    async def background_loop():
        bt.logging.success( 'Starting validator background loop.' )

        block = 0
        while True:

            # Wait for one block.
            bt.logging.success("Background ideling...")
            await asyncio.sleep( bt.__blocktime__ )
            metagraph = subtensor.metagraph(pretrain.NETUID)
            bt.logging.success("End Background step.")
            block += 1

            # Periodically update the weights on the Bittensor blockchain.
            if (block + 1) % 10 == 0:
                # TODO(developer): Define how the validator normalizes scores before setting weights.
                weights = torch.zeros_like( metagraph.S )
                for i in range( len( metagraph.uids ) ):
                    weights[ i ] = scores[ i ] if i in scores else 0.0
                weights = torch.nn.functional.normalize( weights, p=1.0, dim=0)
                bt.logging.success(f"Setting weights: {weights}")
                # This is a crucial step that updates the incentive mechanism on the Bittensor blockchain.
                # Miners with higher scores (or weights) receive a larger share of TAO rewards on this subnet.
                result = subtensor.set_weights(
                    netuid=pretrain.NETUID,  # Subnet to set weights on.
                    wallet=wallet,  # Wallet to sign set weights using hotkey.
                    uids=metagraph.uids,  # Uids of the miners to set weights for.
                    weights=weights,  # Weights to set for the miners.
                    wait_for_inclusion=True,
                )
                if result:
                    bt.logging.success("Successfully set weights.")
                else:
                    bt.logging.error("Failed to set weights.")


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
