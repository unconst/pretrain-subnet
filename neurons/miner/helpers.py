
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

import wandb
import torch
import typing
import pretrain
import bittensor as bt

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


def compute_gradients_on_model( 
        self,
        model: torch.nn.Module,
        batch_size: int,
        sequence_length: int,
        pages: typing.List[int],
    ) -> typing.Dict[str, torch.Tensor]:
    """
        Computes the gradients of the given model on a subset of data.

        This function initializes a data loader with specified parameters, sets the model to training mode, and iterates
        through the dataset to compute gradients. It also manages CUDA's benchmarking mode and clears GPU cache to avoid
        potential out-of-memory errors. The computed gradients are averaged over all batches and returned.

        Parameters:
            model (torch.nn.Module): The model on which to compute gradients.
            batch_size (int): The batch size for the data loader.
            sequence_length (int): The sequence length for the data loader.
            pages (typing.List[int]): List of page indices to be loaded by the data loader.

        Returns:
            typing.Dict[str, torch.Tensor]: A dictionary mapping parameter names to their computed gradients.

        Usage:
            >>> model = SomeTorchModel()
            >>> batch_size = 32
            >>> sequence_length = 128
            >>> pages = [1, 2, 3, 4]
            >>> gradients = await compute_gradients_on_model (model,  batch_size, sequence_length, pages )
    """
    # Initialize the dataloader.
    loader = pretrain.dataset.SubsetFalconLoader(
        batch_size = batch_size,
        sequence_length = sequence_length,
        pages = pages
    )
    bt.logging.success( f'Loaded data subset.' )
    # Enable CUDA's benchmarking mode.
    # This optimizes CUDA's algorithm choices based on input shapes. 
    # NOTE: This should be enabled if input sizes are consistent across batches. Otherwise, 
    # it might have a negative performance impact due to continuous algorithm recalculations.
    torch.backends.cudnn.benchmark = True
    # Reset any previously calculated gradients
    model.zero_grad()
    # Set the model in training mode (this affects layers like dropout and batchnorm)
    model.train()
    bt.logging.success( f'Started gradient computation.' )
    # Iterate over samples this ends once the loader runs out.
    average_loss = 0.0
    n_tokens = 0.0
    n_examples = 0.0
    n_batches = 0.0
    for i, batch in enumerate( loader ):
        # Move the batch to the same device as the model
        inputs = batch.to( model.device )
        # The following block shouldn't have 'torch.no_grad()'. 
        # NOTE: The forward pass needs to build a computation graph for backward.
        # Wrapping it in 'no_grad()' would prevent gradient computation.
        outputs = model( inputs, labels=inputs )
        # Compute gradients based on the loss
        outputs.loss.backward()
        # Remove the computational graph from the loss to save memory
        average_loss += outputs.loss.detach().item()
        # Clear GPU cache to free up memory and avoid potential CUDA out-of-memory errors
        torch.cuda.empty_cache()
        bt.logging.success( f'Acc: step: {i} loss: {outputs.loss}' )
        n_tokens += inputs.numel()
        n_examples += inputs.shape[0]
        n_batches += 1
    # Serialize the averaged gradients into the synapse object
    with torch.no_grad():
        grads = { k: v.grad / (i+1) for k, v in model.named_parameters() if v.grad is not None}
    # Return gradients.
    bt.logging.success( f'Finished gradient computation.' )
    return grads, average_loss/(i+1), n_tokens, n_examples, n_batches

