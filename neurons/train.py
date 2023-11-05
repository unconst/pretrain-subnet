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
import torch
import random
import argparse
import pretrain
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

    # Set the number of epochs
    parser.add_argument("--num_epochs", type=int, default=1, help="Number of training epochs")

    # Set the number of pages trained per epoch
    parser.add_argument("--pages_per_epoch", type=int, default=1, help="Number of pages trained on per epoch")

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.logging.add_args(parser)

    # Parse the arguments and create a configuration namespace
    config = bt.config(parser)

    # Expand the user path and create a full path for the model
    config.full_path = os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            config.logging.logging_dir,
            config.wallet.name,
            config.wallet.hotkey,
            pretrain.NETUID,
            "miner",
        )
    )

    # Set the default model path if it wasn't provided in the command line
    if config.model_path == None:
        config.model_path = config.full_path + '/' + 'model.pth'

    # Create the directory for the model path if it does not exist
    if not os.path.exists(os.path.dirname(config.model_path)):
        os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    return config

# Parse the configuration
config = get_config()

# Print the entire configuration setup
print(config)

# Initialize and configure the model for pretraining
model = pretrain.model.get_model()  # Get the model from the pretrain module
model.zero_grad()  # Reset gradients to zero
model.train()  # Set the model to training mode
model.to(config.device)  # Move the model to the specified device

# Initialize the optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=0.000001, weight_decay=0.01)

import random

# Initialize a variable to keep track of the best average loss
best_avg_loss = float('inf')

# Start the training loop
for epoch in range(config.num_epochs):
    # Initialize loss accumulator for the epoch
    epoch_loss = 0.0

    # Prepare the data loader with random pages for each epoch
    bt.logging.success( f"Loading {config.pages_per_epoch} pages for training this epoch" )
    random_pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range( config.pages_per_epoch )]
    loader = pretrain.dataset.SubsetFalconLoader(batch_size=pretrain.batch_size, sequence_length=pretrain.sequence_length, pages=random_pages)

    # Enumerate over the data loader
    n_batches = 0
    for i, batch in enumerate(loader):
        # Move the input batch to the device
        inputs = batch.to(model.device)
        
        # Forward pass: compute the model output and loss
        outputs = model(inputs, labels=inputs)
        
        # Accumulate the loss for the epoch
        epoch_loss += outputs.loss.item()
        
        # Backward pass: compute the gradient of the loss with respect to model parameters
        outputs.loss.backward()
        
        # Clear the memory cache to avoid CUDA out of memory issues
        torch.cuda.empty_cache()
        
        # Update model parameters
        optimizer.step()
        
        # Log the loss for the current step
        n_batches += 1
        bt.logging.success(f'Step: {i} loss: {outputs.loss.item()}')

    # Calculate the average loss for the epoch
    avg_loss = epoch_loss / n_batches
    
    # Log the average loss for the epoch
    bt.logging.success(f'Epoch: {epoch} average loss: {avg_loss}')

    # Check if the average loss of this epoch is the best we've seen so far
    if avg_loss < best_avg_loss:
        best_avg_loss = avg_loss  # Update the best average loss
        bt.logging.success(f'New best average loss: {best_avg_loss}. Saving model...')
        
        # Save the model state to the specified path
        torch.save(model.state_dict(), config.model_path)

