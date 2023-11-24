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

# Tools for performing validation over models.

import math
import torch
import typing
import pretrain
import traceback
import bittensor as bt

def better(i, j, p, b, losses, metadata):
    """
    Determines the winner between two models based on the epsilon adjusted loss.

    Parameters:
        i (int): Index of the first model.
        j (int): Index of the second model.
        p (int): Page index.
        b (int): Batch index.
        losses (dict): A dictionary containing loss values.
        metadata (dict): A dictionary containing metadata of models.

    Returns:
        bool: True if model i is better, False otherwise.
    """
    # Retrieve loss values for both models
    il = losses[i][p][b]
    jl = losses[j][p][b]

    # Return false if the timestamp is missing in metadata for model i
    if 'timestamp' not in metadata[i]: 
        return False

    # Return true if the timestamp is missing in metadata for model j
    if 'timestamp' not in metadata[j]: 
        return True 

    # Retrieve timestamps from metadata
    it = metadata[i]['timestamp']
    jt = metadata[j]['timestamp']

    # Adjust loss based on timestamp and pretrain epsilon
    il = (1 - pretrain.timestamp_epsilon) * il if it < jt else il
    jl = (1 - pretrain.timestamp_epsilon) * jl if jt < it else jl

    # Compare adjusted losses to determine the better model
    return il < jl

def compute_wins(losses, batches, metadata):
    """
    Computes the wins and win rate for each model based on loss comparison.

    Parameters:
        losses (dict): A dictionary of losses for each model.
        batches (dict): A dictionary of data batches.
        metadata (dict): A dictionary containing metadata of models.

    Returns:
        tuple: A tuple containing two dictionaries, one for wins and one for win rates.
    """
    uids = losses.keys()
    wins = {uid: 0 for uid in uids}
    win_rate = {uid: 0 for uid in uids}

    # Iterate over each pair of models
    for i in uids:
        total_matches = 0
        for j in uids:
            if i == j: 
                continue
            for p in batches.keys():
                for b, _ in enumerate(batches[p]):
                    # Increment wins for model i if it's better than model j
                    wins[i] += 1 if better(i, j, p, b, losses=losses, metadata=metadata) else 0
                    total_matches += 1

        # Calculate win rate for model i
        win_rate[i] = wins[i] / total_matches if total_matches > 0 else 0

    return wins, win_rate

def compute_losses(model, batches: typing.Dict[int, typing.List[torch.Tensor]], device: str) -> typing.Dict[int, typing.List[float]]:
    """
    Computes the losses for a given model on provided batches.

    Parameters:
        model (torch.nn.Module): The model for which losses are to be computed.
        batches (dict): A dictionary of data batches, indexed by page numbers.
        device (str): The device to use for computation (e.g., 'cpu', 'gpu').

    Returns:
        dict: A dictionary with page indices as keys and lists of loss values as values.
    """
    model.to(device)
    model.eval()
    losses_per_page = {}  # Initialize dictionary to store loss values

    # Iterate over each page and corresponding batches
    for page, batch_list in batches.items():
        page_losses = []  # List to store losses for the current page

        for batch in batch_list:
            try:
                inputs = batch.to(device)
                outputs = model(inputs, labels=inputs)
                loss = outputs.loss.item()  # Extract scalar loss value
                page_losses.append(loss)
            except Exception as e:
                bt.logging.error(f"Exception occurred: {e}")
                traceback.print_exc()  # Print the stack trace
                page_losses.append(math.inf)  # Use infinity to indicate failure

        losses_per_page[page] = page_losses

    return losses_per_page
