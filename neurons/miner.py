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
import wandb
import torch
import string
import random
import argparse
import pretrain
import pretrain as pt
import bittensor as bt

# === Config ===
def get_config():
    """
    Set up and parse the command-line arguments to configure the system.

    The configuration is responsible for setting up the environment including
    the model path, device to use, and the bittensor wallet and logging configurations.

    Returns:
        A namespace object containing the configuration parameters.
    """

    # Initialize an argument parser
    parser = argparse.ArgumentParser()

    # Add model_path argument which allows the user to specify the path of the model
    parser.add_argument("--model_path", type=str, required=False, help="Override model path")

    # Add device argument which defaults to 'cuda' if available, else 'cpu'
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")

    # Add device argument which defaults to 'cuda' if available, else 'cpu'
    parser.add_argument("--load_best", action='store_true', help='If set, the miner loads the best model from wandb to train off.' ) 

    # Add device argument which defaults to 'cuda' if available, else 'cpu'
    parser.add_argument("--load_uid", type=int, default=None, help='If passed loads the model under the specified uid.' )  

    # Set the number of epochs
    parser.add_argument("--num_epochs", type = int, default = -1, help="Number of training epochs (-1 is infinite)")

    # Training lr.
    parser.add_argument("--lr", type = float, default = 0.00001, help="Learning rate.")

    # Training batch size
    parser.add_argument("--bs", type = int, default = pretrain.batch_size, help="Batch size")

    # Training sequence length
    parser.add_argument("--sl", type = int, default = pretrain.sequence_length, help="Sequence length")

    # Training accumulation steps per step.
    parser.add_argument("--accumulation_steps", type = int, default = 5, help="The number of training accumulation steps.")

    # Set the number of pages trained per epoch
    parser.add_argument("--pages_per_epoch", type = int, default=10, help="Number of pages trained on per epoch")

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)

    return config

# Parse and print configuration
config = get_config()
print(config)

# Create bittensor objects.
bt.logging( config = config )
wallet = bt.wallet( config = config ) 
subtensor = bt.subtensor( config = config )
metagraph = subtensor.metagraph( pretrain.NETUID )

# Initialize the model based on the best on the network.
if config.load_best:
    model = pt.graph.best_model( metagraph, device = config.device )

# Initialize the model based on a passed uid.
elif config.load_uid is not None:
    # Sync the state from the passed uid.
    pt.graph.sync( config.load_uid, metagraph )
    model = pt.graph.model( config.load_uid, device = config.device )

# Initialize the model from scratch.
else:
    model = pt.model.get_model()

# Init model.
model.train() 
model.to( config.device ) 
optimizer = torch.optim.AdamW( model.parameters(), lr = config.lr, weight_decay=0.01)

# Initialize pretraining wandb run from wallet.
wandb_run = pt.mining.init( 
    wallet, 
    metagraph = metagraph 
)

# Save the initial model state.
pt.mining.push( wallet, model, wandb_run )

# Start the training loop
epoch_step = 0
global_step = 0
n_acc_steps = 0
accumulation_steps = config.accumulation_steps  

try:
    while epoch_step < config.num_epochs or config.num_epochs == -1:
        # Initialize loss accumulator for the epoch
        epoch_loss = 0.0

        # Prepare the data loader with random pages for each epoch
        bt.logging.success( f"Loading {config.pages_per_epoch} pages for training this epoch" )
        random_pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range( config.pages_per_epoch )]
        loader = pretrain.dataset.SubsetFalconLoader(
            batch_size = config.bs, 
            sequence_length = config.sl, 
            pages = random_pages
        )

        # Enumerate over the data loader
        n_batches = 0
        optimizer.zero_grad()  # Initialize gradients to zero

        for i, batch in enumerate(loader):
            # Move the input batch to the device
            inputs = batch.to(model.device)
            
            # Forward pass: compute the model output and loss
            outputs = model(inputs, labels=inputs)

            loss = outputs.loss / accumulation_steps  # Scale loss
            loss.backward()  # Accumulate gradients

            if (i + 1) % accumulation_steps == 0:
                n_acc_steps += 1
                optimizer.step()  # Perform a single optimization step
                optimizer.zero_grad()  # Clear gradients
                bt.logging.success(f'Step: {n_acc_steps} loss: {outputs.loss.detach().item()}')
                wandb_run.log( { 'loss': outputs.loss.detach(), 'n_batches': n_batches }, step = n_acc_steps )

            torch.cuda.empty_cache()
                        
            # Log the loss for the current step
            n_batches += 1
            global_step += 1
            epoch_loss += outputs.loss.detach().item()

        # Calculate the average loss for the epoch
        avg_loss = epoch_loss / n_batches

        # Log the average loss for the epoch
        bt.logging.success(f'Epoch: {epoch_step} average loss: {avg_loss}')
        epoch_step += 1

        # Check if the average loss of this epoch is the best we've seen so far
        if avg_loss < best_avg_loss * ( 1 - pretrain.timestamp_epsilon ):
            best_avg_loss = avg_loss  # Update the best average loss
            bt.logging.success(f'New best average loss: {best_avg_loss}. Saving model...')
            # Push the model to my wandb run.
            pt.mining.push( wallet, model, wandb_run )

finally: 
    wandb.finish()