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
    parser.add_argument("--load_run_id", type=str, default=None, help='If passed loads the model under this run id' )  

    # Add device argument which defaults to 'cuda' if available, else 'cpu'
    parser.add_argument("--continue_id", type=str, default=None, help='If passed continues from the model on the passed run.' )  

    # Set the number of epochs
    parser.add_argument("--num_epochs", type = int, default = -1, help="Number of training epochs (-1 is infinite)")

    # Training lr.
    parser.add_argument("--lr", type = float, default = 1e-7, help="Learning rate.")

    # Training batch size
    parser.add_argument("--bs", type = int, default = pretrain.batch_size, help="Batch size")

    # Training sequence length
    parser.add_argument("--sl", type = int, default = pretrain.sequence_length, help="Sequence length")

    # Set the number of pages trained per epoch
    parser.add_argument("--pages_per_epoch", type = int, default=5, help="Number of pages trained on per epoch")

    # Include wallet and logging arguments from bittensor
    bt.wallet.add_args(parser)
    bt.subtensor.add_args(parser)
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

# Create bittensor objects and check uid.
bt.logging( config = config )
wallet = bt.wallet( config = config ) 
subtensor = bt.subtensor( config = config )
metagraph = subtensor.metagraph( pretrain.NETUID )
if wallet.hotkey.ss58_address not in metagraph.hotkeys: 
    bt.logging.error("You are not registered. Use `btcli s recycle_register` to register.")
    exit()
my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
bt.logging.success( f'You are registered with address: {wallet.hotkey.ss58_address} and uid: {my_uid}' )

# Initialize and configure the model for pretraining
model = pretrain.model.get_model()  # Get the model from the pretrain module
torch.save(model.state_dict(), config.model_path)


def get_run_from_id( run_id ):
    run_path = f"opentensor-dev/openpretraining/{run_id}"
    bt.logging.success(f'Loading model from path: {run_path}')
    api = wandb.Api( timeout = 100 )
    return api.run(run_path)

# Optionall load the model from the passed run id:
def load_model_from_run( run ):
    model_file = run.file("model.pth")
    model_file.download(replace=True, root = os.path.dirname(config.model_path) )
    bt.logging.success(f'Loaded and saved model to: {config.model_path}')

# Model is pulled from specific run.
if config.load_run_id != None:
    bt.logging.success(f'Loading based on --config.load_run_id {config.model_path}')
    load_model_from_run( get_run_from_id( config.load_run_id ) )
    
# Model is pulled from best on network
elif config.load_best:
    bt.logging.success(f'Loading based on --config.load_best')
    all_valid_runs = pretrain.get_miner_runs( metagraph )
    sorted_valid_runs = sorted( list( all_valid_runs.values()), key=lambda x: x['incentive'])
    load_model_from_run( get_run_from_id(sorted_valid_runs[0]['run']) )

elif config.continue_id:
    run = get_run_from_id( config.continue_id  )
    run_hotkey = run.config['hotkey']
    load_model_from_run( run )

# Model is reinited fresh.
else:
    bt.logging.success(f'Starting model from scratch')

# Load the model.
model_weights = torch.load( config.model_path, map_location=torch.device(config.device) )
model.load_state_dict( model_weights )
model.zero_grad()  # Reset gradients to zero
model.train()  # Set the model to training mode
model.to(config.device)  # Move the model to the specified device

# Initialize the optimizer
optimizer = torch.optim.AdamW( model.parameters(), lr = config.lr, weight_decay=0.01)

import random

# Initialize a variable to keep track of the best average loss
best_avg_loss = float('inf')

# Initialize your wandb run
# NOTE: removing the "miner-" from this line will mean your miner is not picked up by validators.
run_name = f'miner-{my_uid}-' + ''.join(random.choice( string.ascii_uppercase + string.digits ) for i in range(10))
config.uid = my_uid
config.hotkey = wallet.hotkey.ss58_address
config.run_name = run_name
config.version = pretrain.__version__
config.type = 'miner'
if config.continue_id:
    # Attempts to continue run from previous id.
    bt.logging.success(f'Continuing wandb run from id {config.continue_id}')
    wandb_run = wandb.init(
        id = config.continue_id,
        name = run_name,
        anonymous = "allow",
        resume = "must",
        project = 'openpretraining',
        entity = 'opentensor-dev',
        config = config,
        dir = config.full_path,
        allow_val_change=True,
    )
else:
    bt.logging.success(f'Starting fresh wandb run')
    wandb_run = wandb.init(
        name = run_name,
        anonymous = "allow",
        reinit = False,
        project = 'openpretraining',
        entity = 'opentensor-dev',
        config = config,
        dir = config.full_path,
    )
bt.logging.success(f'\n\nSuccessfully started your wandb run {wandb_run.id}, you can continue it at a lated data by passing --continue_id {wandb_run.id}\n\n')

# Signature
signature = wallet.hotkey.sign( wandb_run.id.encode() ).hex()
config.signature = signature
wandb.config.update( config, allow_val_change=True )
bt.logging.success(f'Successfully signed wandb run with signature {config.signature}')

# Save the model to wandb.
wandb.save( config.model_path )
bt.logging.success('Pushed artifact to the wandb run.')

# Start the training loop
epoch_step = 0
global_step = 0
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
    for i, batch in enumerate(loader):

        # Move the input batch to the device
        inputs = batch.to(model.device)
        
        # Forward pass: compute the model output and loss
        outputs = model(inputs, labels=inputs)
               
        # Backward pass: compute the gradient of the loss with respect to model parameters
        outputs.loss.backward()
        
        # Clear the memory cache to avoid CUDA out of memory issues
        torch.cuda.empty_cache()
        
        # Update model parameters
        optimizer.step()
        
        # Step loss
        wandb.log( { 'loss': outputs.loss.detach(), 'n_batches': n_batches }, step = global_step )
        
        # Log the loss for the current step
        n_batches += 1
        global_step += 1
        epoch_loss += outputs.loss.detach().item()
        bt.logging.success(f'Step: {i} loss: {outputs.loss.detach().item()}')

    # Calculate the average loss for the epoch
    avg_loss = epoch_loss / n_batches

    # Log the average loss for the epoch
    bt.logging.success(f'Epoch: {epoch_step} average loss: {avg_loss}')
    epoch_step += 1

    # Check if the average loss of this epoch is the best we've seen so far
    if avg_loss < best_avg_loss:
        best_avg_loss = avg_loss  # Update the best average loss
        bt.logging.success(f'New best average loss: {best_avg_loss}. Saving model...')
        
        # Save the model state to the specified path
        torch.save( model.state_dict(), config.model_path )

        # Save the new best model to wandb.
        wandb.save( config.model_path )
        bt.logging.success('Pushed the new artifact to the wandb run.')