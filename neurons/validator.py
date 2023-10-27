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

# Step 1: Import necessary libraries and modules
import os
import math
import time
import wandb
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
    parser.add_argument( '--n_concurrent_forward', type=int, default=4, help='Number of allowed concurrent foward requests.' )
    parser.add_argument( '--n_concurrent_forward_per_uid', type=int, default=1, help='Number of allowed concurrent foward requests per uid.')
    parser.add_argument( '--batch_size', type=int, default=3, help='Eval batch size' )
    parser.add_argument( '--sequence_length', type=int, default=512, help='Eval sequence length' )
    parser.add_argument( '--device', type = str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to run the miner on.' )
    parser.add_argument( "--wandb.off", action="store_true", help="Turn off wandb.", default=False)
    parser.add_argument( "--wandb.project_name", type=str, help="The name of the project where you are sending the new run.", default="openpretraining" )
    parser.add_argument( "--wandb.entity", type=str, help="An entity is a username or team name where youre sending runs.", default="opentensor-dev" )
    parser.add_argument( "--wandb.offline", action="store_true", help="Runs wandb in offline mode.", default=False,)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    config = bt.config(parser)
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

    # === Logging ===
    bt.logging(config=config, logging_dir=config.full_path)
    bt.logging.info(config)

    # === Bittensor objects ===
    wallet = bt.wallet(config=config)
    subtensor = bt.subtensor(config=config)
    metagraph = subtensor.metagraph(pretrain.NETUID)
    if wallet.hotkey.ss58_address not in metagraph.hotkeys: raise Exception("You are not registered.")
    my_subnet_uid = metagraph.hotkeys.index(wallet.hotkey.ss58_address)
    helpers.init_wandb( wallet = wallet, config = config, type = 'validator', uid = my_subnet_uid )
    bt.logging.info(f"Wallet: {wallet}")
    bt.logging.info(f"Subtensor: {subtensor}")
    bt.logging.info(f"Metagraph: {metagraph}")
    bt.logging.info(f"Running miner on uid: {my_subnet_uid}")

    # === Training objects ===
    scores = {}
    model = pretrain.model.get_model().to( config.device )
    optimizer = AdamW( model.parameters(), lr=config.learning_rate )
    
    # === Locks ===
    gpu_lock = asyncio.Lock()
    model_lock = asyncio.Lock()
    forward_lock = asyncio.Semaphore( config.n_concurrent_forward )
    per_uid_locks = { i: asyncio.Semaphore( config.n_concurrent_forward_per_uid ) for i in range(256) }

    # === Forward Function ===
    async def forward( ):
        """
            The `forward` function performs a forward pass through a randomly selected miner axon,
            computes gradients both locally and through a request to the miner, compares them to score the miner,
            and then applies the received gradients to the local model.

            This function is designed to be asynchronous to allow for concurrent execution, while also implementing
            necessary locks to prevent race conditions and ensure the integrity of the operations.
        """
        wandb_event = {}
        wandb_event['begin_forward'] = time.time()

        # Acquire the forward lock to limit concurrent forward calls.
        async with forward_lock:
            bt.logging.success( 'Started validator forward task.' )

            # Get a random miner axon.
            available_uids = [uid.item() for uid in metagraph.uids if metagraph.axons[uid].is_serving]
            if len(available_uids) == 0: return
            uid = random.choice(available_uids)
            bt.logging.success( f'Selected uid:{uid}' )
            wandb_event['uid'] = uid

            # Acquire the per uid lock to limit concurrent forward calls to the same uid.
            async with per_uid_locks[uid]:

                # Build the query for miner
                pages = [ random.randint(1, 968000015) ]
                synapse = pretrain.protocol.ComputeGradients( 
                    batch_size = config.batch_size,
                    sequence_length = config.sequence_length,
                    pages = pages
                )
                wandb_event['batch_size'] = config.batch_size
                wandb_event['sequence_length'] = config.batch_size
                wandb_event['pages'] = pages
                wandb_event['n_pages'] = len(pages)

                # Get current model state.
                async with model_lock:
                    query_model_state = model.state_dict()

                # Serialize the model state into the synapse object.
                synapse.serialize( state_dict = query_model_state ) 
                bt.logging.success( f'Serialized state.' )

                # Make the forward query.
                wandb_event['query_time'] = time.time()
                dendrite = bt.dendrite( wallet = wallet )
                bt.logging.success( f'Sent request.' )
                response = await dendrite.forward( metagraph.axons[ uid ], synapse, timeout = 60, deserialize = False )
                await dendrite.close_session()
                wandb_event['end_query_time'] = time.time()

                # Check for failures.
                if not response.is_success: 
                    bt.logging.error( f'Response failure.' )
                    wandb_event['success'] = False
                    # Failed requests get a mse increase of 1 added.
                    next_score = -10
                    scores[ uid ] = config.alpha * scores[uid] + ( 1 - config.alpha ) * next_score if uid in scores else config.alpha * next_score
                    wandb_event['next_score'] = next_score
                    wandb_event['current_score'] = scores[ uid ] 
                    wandb_event['end_forward'] = time.time()
                    wandb.log( wandb_event )
                    return 
                
                else:
                    wandb_event['success'] = True

                    bt.logging.success( f'Response success.' )
                    # Deserialize the gradients from the response.
                    grads_dict = response.deserialize()
                    # Score the local model by comparing the gradients generated locally vs 
                    # the gradients generated by the miner.
                    async with gpu_lock:
                        # Run eval.
                        wandb_event['begin_eval'] = time.time()
                        bt.logging.success( f'Aquired GPU space.' )
                        # Load the models weights into a new model and position it on the gpu.
                        eval_model = pretrain.model.get_model()
                        eval_model.load_state_dict( query_model_state )
                        eval_model.to( config.device )
                        bt.logging.success( f'Created eval model on GPU')
                        # Compute the gradients on the model.
                        local_grads, loss, n_tokens, n_examples, n_batches  = helpers.compute_gradients_on_model(
                            model = eval_model,
                            batch_size = config.batch_size,
                            sequence_length = config.sequence_length,
                            pages = pages
                        )
                        bt.logging.success( f'Finished local gradient computation with loss: {loss}' )
                        wandb_event['loss'] = loss
                        wandb_event['n_tokens'] = n_tokens
                        wandb_event['n_examples'] = n_examples
                        wandb_event['n_batches'] = n_batches

                        # Compute MSE
                        alpha = 0.99
                        next_score = -helpers.mse_gradients( local_grads, grads_dict )
                        scores[ uid ] = alpha * scores[uid] + ( 1 - alpha ) * next_score if uid in scores else alpha * next_score
                        bt.logging.success( f'Computed MSE: {next_score}' )
                        bt.logging.success( f'Updated score: {scores[uid]}' )
                        wandb_event['next_score'] = next_score
                        wandb_event['current_score'] = scores[ uid ] 
                        wandb_event['end_eval'] = time.time()

                    # Apply the gradients to the local model.
                    async with model_lock:
                        bt.logging.success( f'Aquired model lock.' )
                        wandb_event['begin_apply_grad'] = time.time()
                        # Accumulate grads on local model.
                        model.to('cpu')
                        for name_j, param_j in model.named_parameters():
                            if name_j in grads_dict and grads_dict[name_j] is not None:
                                param_j.grad = param_j.grad + grads_dict[name_j].to('cpu') if param_j.grad is not None else grads_dict[name_j].to('cpu')
                        bt.logging.success( f'Applied gradients to model.' )

                        # Take a step using the optimizer to update model parameters
                        optimizer.step()
                        bt.logging.success( f'Applied step.' )

                        # Zero out the gradients to free up memory and avoid accidental accumulation in the next iteration
                        model.zero_grad()
                        wandb_event['end_apply_grad'] = time.time()

                    # Finish.
                    bt.logging.success( f'Finished forward.' )
                    wandb_event['end_forward'] = time.time()
                    wandb.log( wandb_event )
                    return 



    # === Background Function ===
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

            # Set weights every 10 blocks.       
            if block % 50 == 0:
                bt.logging.success( f'Setting weights on chain' )

                # Fill weights. weight_i = exp( -score_i ) / SUM_j exp( -score_j )
                # Where score_i is the moving average of negative of the MSE between grads returned and grads computed.
                weights = torch.zeros_like( metagraph.S )
                for i in range( len( metagraph.uids ) ):
                    weights[ i ] = math.exp( scores[ i ] ) if i in scores else 0.0

                # Normalize the scores to 1.0
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

    # === Training loop ===
    async def training_loop():
        bt.logging.success( 'Starting validator training loop.' )
        while True:
            # Create a task for each request and add it to the event loop
            asyncio.create_task( forward() )
            # Wait for a short time before creating the next task
            await asyncio.sleep( 1 )

    # === Main loop ===
    async def main_loop():
        bt.logging.success( 'Starting validator main loop.' )
        asyncio.run( training_loop() )
        asyncio.run( background_loop() )

    # === Start ===
    loop = asyncio.get_event_loop()
    try:
        def handle_exception(loop, context):
            bt.logging.error(f"Caught exception: {context['exception']}")
        # Run the main loop until completion.
        loop.set_exception_handler(handle_exception)
        loop.run_until_complete(main_loop())

    # If we encounter an unexpected error, log it for debugging.
    except RuntimeError as e:
        bt.logging.error(e)
        traceback.print_exc()

    # If the user interrupts the program, gracefully exit.
    except KeyboardInterrupt:
        bt.logging.success("Keyboard interrupt detected. Exiting validator.")
        wandb.finish()
        exit()

# === Init ===
if __name__ == "__main__":
    main( get_config() )
