import math
import torch
import typing
import pretrain
import traceback
import bittensor as bt

# Win function.
# Determines the winner based on the epsilon adjusted loss
# Models that were created earlier have a 3% decrease in loss
def better( i, j, p, b, losses, metadata ):
    il = losses[ i ][ p ][ b ]
    jl = losses[ j ][ p ][ b ]
    if 'timestamp' not in metadata[ i ]: return False 
    if 'timestamp' not in metadata[ j ]: return True 
    it = metadata[ i ]['timestamp']
    jt = metadata[ j ]['timestamp']
    il = (1 - pretrain.timestamp_epsilon) * il if it < jt else il
    jl = (1 - pretrain.timestamp_epsilon) * jl if jt < it else jl
    if il < jl: return True
    else: return False

def compute_wins( 
        losses,
        batches,
        metadata,
    ):
    # Compute wins this step.
    uids = losses.keys()
    wins = { uid: 0 for uid in uids }
    win_rate = { uid: 0 for uid in uids }
    for i in uids:
        total_matches = 0
        for j in uids:
            if i == j: continue
            for p in batches.keys():
                for b, _ in enumerate( batches[ p ] ):
                    wins[ i ] += 1 if better( i, j, p, b, losses = losses, metadata = metadata ) else 0
                    total_matches += 1
        win_rate[ i ] = wins[ i ] / total_matches

    return wins, win_rate

def compute_losses( 
        model, 
        batches: typing.Dict[int, typing.List[torch.Tensor]],
        device: str
    ) -> typing.Dict[int, typing.List[float]]:
    # Initialize a dictionary to store loss values for each page
    losses_per_page = {}

    # Iterate over each page and its corresponding batches
    for page, batches in batches.items():
        page_losses = []  # List to store losses for the current page

        # Process each batch and compute its loss
        for batch in batches:
            try:
                # Perform a forward pass with the model to compute the loss
                inputs = batch.to( device )
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss.item()  # Get the scalar loss value
                page_losses.append(loss)
            except Exception as e:
                # Log the exception and append infinity to indicate failure
                bt.logging.error(f"Exception occurred: {e}")
                traceback.print_exc()  # Correctly print the stack trace
                page_losses.append(math.inf)
        
        # Update the dictionary with the losses for the current page
        losses_per_page[page] = page_losses

    return losses_per_page