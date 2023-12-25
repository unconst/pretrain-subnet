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
import json
import torch
import random
import string
import typing
import warnings
import pretrain as pt
import bittensor as bt
import shutil
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM
from safetensors.torch import load_model, save_model

def path(wallet: int) -> str:
    """
    Constructs a file path for storing wallet-related data.

    Parameters:
    wallet (int): An object representing the wallet.

    Returns:
    str: A string representing the file path.
    """
    return os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            bt.logging.config().logging.logging_dir,
            wallet.name,
            wallet.hotkey_str,
            pt.NETUID,
            "miner",
        )
    )

def model_path(wallet: int) -> str:
    """
    Constructs a file path for storing the model related to a specific wallet.

    Parameters:
    wallet (int): An object representing the wallet.

    Returns:
    str: A string representing the model file path.
    """
    return os.path.expanduser(
        "{}/{}/{}/netuid{}/{}".format(
            bt.logging.config().logging.logging_dir,
            wallet.name,
            wallet.hotkey_str,
            pt.NETUID,
            "miner/model.safe",
        )
    )


def uid(wallet, metagraph: typing.Optional[bt.metagraph] = None) -> typing.Optional[ int ]:
    """
    Retrieves the user ID (UID) based on the wallet and metagraph.

    Parameters:
    wallet: Wallet object containing user credentials.
    metagraph (Optional[bt.metagraph]): Metagraph object for additional network data.

    Returns:
    The UID if available, otherwise None.
    """
    if not metagraph:
        metagraph = bt.subtensor().metagraph(pt.NETUID)
    if wallet.hotkey.ss58_address not in metagraph.hotkeys: 
        bt.logging.error(f"You don't have a UID because your wallet is not registered on subnet 9. Use `btcli s register --netuid {pt.NETUID}` to register.")
        return None
    return metagraph.hotkeys.index(wallet.hotkey.ss58_address)


def push( model, repo_name, token):
    """
    Saves the model state previously saved and updates the huggingface run.

    Parameters:
        model: model to be pushed.
        repo_name: the name of huggingface repo
        token: credential HF access token
    Returns:
        None
    """
    model.push_to_hub(repo_name=repo_name, repo_id=repo_name, token=token, safe_serialization=True)


def is_valid_size(model) -> bool:
    """
    check the size of model
    Parameters:
        model: The pytorch pretrained model

    Returns:
        Bool.
    """
    model_size = sum(p.numel() for p in model.parameters())
    # current size of gpt2 is 122268040, previous size is 57868320.
    # distilgpt2 size is 81912576 try to get a new model size that no one pretrained before
    if not (82_000_000 <= model_size <= 122_200_000):
        warnings.warn("This model size is not Valid, please make sure you model parameter size is between 82M and 122M .")
        return False
    return True


def save( wallet, repo_name ):
    """
    Saves the model state to your wallet path.

    Parameters:
        wallet: Wallet object containing user credentials.
        repo_name: The repo to be saved.

    Returns:
        model .
    """
    _model_path = path(wallet)
    if not os.path.exists(os.path.dirname(_model_path)):
        os.makedirs(os.path.dirname(_model_path), exist_ok=True)
    try:
        # Save the model state to the specified path
        snapshot_download(repo_id=repo_name, \
                        local_dir=_model_path, \
                            local_dir_use_symlinks=False)
        model = AutoModelForCausalLM.from_pretrained(_model_path, use_safetensors=True, local_files_only=True)
        if is_valid_size(model):
            return model
    except:
        shutil.rmtree(_model_path) 
        raise ValueError('Model failed to save.')



def load_from_hf(wallet, repo_name):
    """
    Loads the model state to your wallet path.

    Parameters:
        wallet: Wallet object containing user credentials.
        repo_name: huggingface repo name
    Returns:
        model: model loaded under wallet path.
    """

    model = save(wallet, repo_name)

    if is_valid_size(model):
        return model
    else:
        raise ValueError('Model failed to load.')

def load( wallet, device: str = 'cpu'):
    """
    Loads the model state to your wallet path.

    Parameters:
        wallet: Wallet object containing user credentials.

    Returns:
        model: model loaded under wallet path.
    """
    
    _model_path = path(wallet)
    model = AutoModelForCausalLM.from_pretrained(_model_path, use_safetensors=True, local_files_only=True) 
    model.to(device)
    if is_valid_size(model):
        return model
    else:
        shutil.rmtree(_model_path) 
        raise ValueError('Model failed to load.')

def update( wallet, repo_name, token, uid):
    """
    Update the model in huggingface and this subnet.

    Parameters:
        wallet: Wallet object containing user credentials.
        repo_name: huggingface repo name
        token: credential HF access token
        uid: User ID 
    Returns:
        None
    """
    model = save( wallet,repo_name)
    push( model, repo_name, token, uid)


