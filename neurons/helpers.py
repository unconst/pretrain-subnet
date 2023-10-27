import torch
import typing
import pretrain
import bittensor as bt

def mse_gradients(
        grads_A: typing.Dict[str, torch.Tensor],
        grads_B: typing.Dict[str, torch.Tensor],
    ) -> typing.Dict[str, torch.Tensor]:

    # Ensure the keys in both dictionaries match
    assert set(grads_A.keys()) == set(grads_B.keys()), "Mismatched keys between grads_A and grads_B"

    # Iterate through each key in grads_A (keys in grads_B should match due to the assert statement)
    mse_sum = 0
    for key in grads_A.keys():
        # Compute the squared difference between corresponding gradients
        squared_diff = (grads_A[key].to('cpu') - grads_B[key].to('cpu')) ** 2
        
        # Compute the mean of the squared difference
        mse_sum += torch.mean(squared_diff).detach().item()

    # Return sum.
    return mse_sum
      
def compute_gradients_on_model( 
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
            outputs.loss.detach()
            # Clear GPU cache to free up memory and avoid potential CUDA out-of-memory errors
            torch.cuda.empty_cache()
            bt.logging.success( f'Acc: step: {i} loss: {outputs.loss}' )
        # Serialize the averaged gradients into the synapse object
        with torch.no_grad():
            grads = { k: v.grad / (i+1) for k, v in model.named_parameters() if v.grad is not None}
        # Return gradients.
        bt.logging.success( f'Finished gradient computation.' )
        return grads