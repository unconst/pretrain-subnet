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
    parser.add_argument( '--batch_size', type=int, default=8, help='Eval batch size' )
    parser.add_argument( '--sequence_length', type=int, default=512, help='Eval sequence length' )
    parser.add_argument( '--n_steps_per_worker', default=1, type=int, help='Number of steps per worker.' )
    parser.add_argument( '--n_eval_steps', default=1, type=int, help='Number of eval steps.' )
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
    model.train()
    
    # Step 8: Build optimizer for validation.
    optimizer = AdamW( model.parameters(), lr=config.learning_rate )
    def apply_grads_to_model_and_step( grads: typing.List[ typing.Dict[ str, torch.Tensor ]] ):
        # Sum grads from all workers on master model.
        model.zero_grad()
        if not isinstance( grads, list ): grads = [ grads ] 
        for grad_dict in grads:
            for (name_j, param_j) in model.named_parameters():
                if name_j in grad_dict:
                    if grad_dict[name_j] is not None:
                        grad_ij = grad_dict[name_j]
                        if param_j.grad is None:
                            param_j.grad = grad_ij.to( config.device )
                        else:
                            param_j.grad += grad_ij.to( config.device )
                    else:
                        bt.logging.trace(f'remote_grads[{name_j}] is None')
                else:
                    bt.logging.trace(f'{name_j} not in remote_grads:')

        # Average grads based on number of workers.
        for (_, param_j) in model.named_parameters():
            if param_j.grad is not None:
                param_j.grad /= len( grads )

        # Step.
        optimizer.step()
        model.zero_grad()

    # Function for computing loss on a subset of the dataset.
    dataloader = pretrain.dataset.get_dataloader( config.batch_size, config.sequence_length )
    def compute_current_loss_on_subset( n_steps ):
        step = 0
        loss = 0
        while True:
            batch = next( dataloader )
            inputs = batch.to( config.device )            
            outputs = model( inputs, labels = inputs )
            loss += outputs.loss      
            if step >= n_steps: break
            else: step += 1
        return loss.item()/n_steps
    
    def get_random_available_miner_axon( ) -> typing.Optional[int]:
        available_uids = [uid.item() for uid in metagraph.uids if metagraph.active[ uid ].item() == 1 and metagraph.axons[ uid ].is_serving ]
        if len( available_uids ) == 0: return None
        random_miner_uid = random.choice( available_uids )
        return random_miner_uid

    # Step 7: The Main Validation Loop
    bt.logging.info("Starting validator loop.")
    step = 0
    prev_loss = compute_current_loss_on_subset( n_steps = config.n_eval_steps )
    bt.logging.success(f'Step: {step} Training loss: {prev_loss}')
    while True:
        try:
            # Select a random uid to query.
            random_miner_uid = get_random_available_miner_axon()
            if random_miner_uid == None: bt.logging.info('No available miners, continuing'); continue
            random_miner_axon = metagraph.axons[ random_miner_uid ]

            # Build the query.
            synapse = pretrain.protocol.ComputeGradients( n_steps = config.n_steps_per_worker )
            synapse.serialize( state_dict = model.state_dict() ) 

            # Make the broadcast query
            dendrite = bt.dendrite( wallet = wallet )
            grads = dendrite.query( random_miner_axon, synapse, timeout = -1, deserialize = True )
            asyncio.get_event_loop().run_until_complete(dendrite.close_session())

            # Apply grads to model and step.
            apply_grads_to_model_and_step( grads )

            # Compute current loss on subset of dataset.
            loss = compute_current_loss_on_subset( n_steps = config.n_eval_steps )
            bt.logging.success(f'Step: {step} Training loss: {loss}')

            # Compute miner score based on previous loss
            score_for_miner = prev_loss - loss
            prev_loss = loss

            # Update score for miner.
            prev_score = scores[ random_miner_uid ] if random_miner_uid < len( scores ) else 0.0
            scores[ random_miner_uid ] = config.alpha * prev_score + (1 - config.alpha) * score_for_miner
            bt.logging.info(f"Scores: {scores}")

            # Periodically update the weights on the Bittensor blockchain.
            if (step + 1) % 10 == 0:
                # TODO(developer): Define how the validator normalizes scores before setting weights.
                weights = torch.zeros_like( metagraph.S )
                for i in range( len( metagraph.uids ) ):
                    weights[ i ] = scores[ i ] if i in scores else 0.0
                weights = torch.nn.functional.normalize( weights, p=1.0, dim=0)
                bt.logging.info(f"Setting weights: {weights}")
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

            # End the current step and prepare for the next iteration.
            step += 1
            # Resync our local state with the latest state from the blockchain.
            metagraph = subtensor.metagraph(pretrain.NETUID)
            # Sleep for a duration equivalent to the block time (i.e., time between successive blocks).
            time.sleep(bt.__blocktime__)

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
