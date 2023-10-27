import torch
import typing
import pretrain

async def mse_gradients(
        grads_A: typing.Dict[str, torch.Tensor],
        grads_B: typing.Dict[str, torch.Tensor],
    ) -> typing.Dict[str, torch.Tensor]:

    # Ensure the keys in both dictionaries match
    assert set(grads_A.keys()) == set(grads_B.keys()), "Mismatched keys between grads_A and grads_B"

    # Initialize an empty dictionary to store the MSE of gradients
    gradients_mse = {}

    # Iterate through each key in grads_A (keys in grads_B should match due to the assert statement)
    for key in grads_A.keys():
        # Compute the squared difference between corresponding gradients
        squared_diff = (grads_A[key] - grads_B[key]) ** 2
        
        # Compute the mean of the squared difference
        mse = torch.mean(squared_diff)
        
        # Store the mse in the gradients_mse dictionary
        gradients_mse[key] = mse

    return gradients_mse
      
async def compute_gradients_on_model( 
        model: torch.nn.Module,
        batch_size: int,
        sequence_length: int,
        pages: typing.List[int],
    ) -> typing.Dict[str, torch.Tensor]:
        # Initialize the dataloader.
        loader = pretrain.dataset.SubsetFalconLoader(
            batch_size = batch_size,
            sequence_length = sequence_length,
            pages = pages
        )
        # Enable CUDA's benchmarking mode.
        # This optimizes CUDA's algorithm choices based on input shapes. 
        # NOTE: This should be enabled if input sizes are consistent across batches. Otherwise, 
        # it might have a negative performance impact due to continuous algorithm recalculations.
        torch.backends.cudnn.benchmark = True
        # Reset any previously calculated gradients
        model.zero_grad()
        # Set the model in training mode (this affects layers like dropout and batchnorm)
        model.train()
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
        # Serialize the averaged gradients into the synapse object
        with torch.no_grad():
            grads = { k: v.grad / (i+1) for k, v in model.named_parameters() if v.grad is not None}
        # Return gradients.
        return grads