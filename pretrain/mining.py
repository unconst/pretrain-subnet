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
import wandb
import random
import string
import typing
import warnings
import pretrain as pt
import bittensor as bt
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

def model_architecture_path(wallet: int) -> str:
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
            "miner/model_architecture.pth",
        )
    )


def runidpath(wallet) -> str:
    """
    Constructs a file path for storing the run ID.

    Parameters:
    wallet: Wallet object containing user credentials.

    Returns:
    str: A string representing the run ID file path.
    """
    return path(wallet) + '/run.json'

def load_runid(wallet: int) -> typing.Optional[str]:
    """
    Retrieves the run ID from the file system.

    Parameters:
    wallet (int): An object representing the wallet.

    Returns:
    Optional[str]: The run ID if available, otherwise None.
    """
    try:
        with open(runidpath(wallet), 'r') as f:
            return json.load(f)['WANDB_RUN_ID']
    except Exception as e:
        return None
    
def find_runid(wallet, metagraph: typing.Optional[bt.metagraph] = None):
    """
    Retrieves the run ID from the file system.

    Parameters:
    wallet (int): An object representing the wallet.

    Returns:
    Optional[str]: The run ID if available, otherwise None.
    """
    if not metagraph:
        metagraph = bt.subtensor().metagraph(pt.NETUID)
    my_uid = uid(wallet, metagraph)
    api = wandb.Api(timeout=100)
    expected_hotkey = metagraph.hotkeys[my_uid]

    # Retrieve runs from the wandb project for this uid
    runs = api.runs(
        f"opentensor-dev/{pt.WANDB_PROJECT}",
        filters={
            "config.version": pt.__version__,
            "config.type": "miner",
            "config.run_name": {
                "$regex": f"miner-{my_uid}-.*"
            },
            "config.hotkey": expected_hotkey,
        }
    )

    # Iterate through runs. Newer runs are processed first.
    for run in runs:
        pt.graph.check_run_validity(run, metagraph)
        return run.id
    return None
    
def save_runid(wallet, run_id):
    with open(runidpath(wallet), 'w') as f:
        json.dump({'WANDB_RUN_ID': run_id}, f)

def new_runid(wallet):
    """
    Generates and stores a new run ID.

    Parameters:
    wallet: Wallet object containing user credentials.

    Returns:
    str: The newly generated run ID.
    """
    run_id = wandb.util.generate_id()
    save_runid( wallet, run_id )
    return run_id


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

def init(wallet, metagraph: typing.Optional[bt.metagraph] = None):
    """
    Initializes the mining process wandb run by setting up necessary directories, wandb run, and configurations.

    Parameters:
        wallet: Wallet object containing user credentials.
        metagraph (Optional[bt.metagraph]): Metagraph object for additional network data.

    Returns:
        wandb_run: Returns a wandb run object if initialization is successful, otherwise None.
    """

    # Create the directory for the model if it does not exist
    mpath = model_path(wallet)
    if not os.path.exists(os.path.dirname(mpath)):
        os.makedirs(os.path.dirname(mpath), exist_ok=True)

    # Get the metagraph if not provided
    if not metagraph:
        metagraph = bt.subtensor().metagraph(pt.NETUID)

    # Find the UID for this wallet.
    my_uid = uid(wallet, metagraph)
    if my_uid is None: 
        return None

    # Try to load the run ID from the file system.
    if load_runid(wallet) is None:
        # If we cant load the runid, try to load the run ID from wandb based on wallet.
        if find_runid(wallet, metagraph) is None:
            save_runid(wallet, new_runid(wallet) )

    # Load the run ID from the file system.
    _run_id = load_runid(wallet)
    
    # Creating a unique run name
    run_name = f'miner-{my_uid}-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    config = bt.config()
    config.uid = my_uid
    config.hotkey = wallet.hotkey.ss58_address
    config.run_name = run_name
    config.version = pt.__version__
    config.type = 'miner'

    # Initialize wandb run
    wandb_run = wandb.init(
        id = _run_id,
        name=run_name,
        anonymous="allow",
        project=pt.WANDB_PROJECT,
        entity='opentensor-dev',
        config=config,
        dir=config.full_path,
        allow_val_change=True,
    )

    # Creating a signature for security
    signature = wallet.hotkey.sign(wandb_run.id.encode()).hex()
    config.signature = signature
    wandb.config.update(config, allow_val_change=True)

    return wandb_run


def init_validator(wallet, metagraph: typing.Optional[bt.metagraph] = None):
    """
    Initializes the validator process wandb run by setting up necessary directories, wandb run, and configurations.

    Parameters:
        wallet: Wallet object containing user credentials.
        metagraph (Optional[bt.metagraph]): Metagraph object for additional network data.

    Returns:
        wandb_run: Returns a wandb run object if initialization is successful, otherwise None.
    """

    # Create the directory for the model if it does not exist
    mpath = model_path(wallet)
    if not os.path.exists(os.path.dirname(mpath)):
        os.makedirs(os.path.dirname(mpath), exist_ok=True)

    # Get the metagraph if not provided
    if not metagraph:
        metagraph = bt.subtensor().metagraph(pt.NETUID)

    # Find the UID for this wallet.
    my_uid = uid(wallet, metagraph)
    if my_uid is None: 
        return None

    # Try to load the run ID from the file system.
    if load_runid(wallet) is None:
        # If we cant load the runid, try to load the run ID from wandb based on wallet.
        if find_runid(wallet, metagraph) is None:
            save_runid(wallet, new_runid(wallet) )

    # Load the run ID from the file system.
    _run_id = load_runid(wallet)
    
    # Creating a unique run name
    run_name = f'validator-{my_uid}-' + ''.join(random.choices(string.ascii_uppercase + string.digits, k=10))
    config = bt.config()
    config.uid = my_uid
    config.hotkey = wallet.hotkey.ss58_address
    config.run_name = run_name
    config.version = pt.__version__
    config.type = 'validator'

    # Initialize wandb run
    wandb_run = wandb.init(
        id = _run_id,
        name=run_name,
        anonymous="allow",
        project=pt.WANDB_PROJECT,
        entity='opentensor-dev',
        config=config,
        dir=config.full_path,
        allow_val_change=True,
    )

    # Creating a signature for security
    signature = wallet.hotkey.sign(wandb_run.id.encode()).hex()
    config.signature = signature
    wandb.config.update(config, allow_val_change=True)

    return wandb_run

def load_run(wallet: int, metagraph: typing.Optional[bt.metagraph] = None) -> typing.Optional['wandb.run']:
    """
    Loads a wandb run based on the wallet and metagraph information.

    Parameters:
    wallet (int): Wallet object containing user credentials.
    metagraph (Optional[bt.metagraph]): Metagraph object for additional network data.

    Returns:
    Optional[wandb.run]: A wandb run object if found, otherwise None.
    """
    if not metagraph:
        metagraph = bt.subtensor().metagraph(pt.NETUID)
    my_uid = uid(wallet, metagraph)
    api = wandb.Api(timeout=100)
    expected_hotkey = metagraph.hotkeys[my_uid]

    # Retrieve runs from the wandb project for this uid
    runs = api.runs(
        f"opentensor-dev/{pt.WANDB_PROJECT}",
        filters={
            "config.version": pt.__version__,
            "config.type": "miner",
            "config.run_name": {
                "$regex": f"miner-{my_uid}-.*"
            },
            "config.hotkey": expected_hotkey,
        }
    )

    # Iterate through runs. Newer runs are processed first.
    for run in runs:
        pt.graph.check_run_validity(run, metagraph)
        return run
    return None

def push( model, repo_name, token, uid):
    """
    Saves the model state previously saved and updates the huggingface run.

    Parameters:
        wallet: Wallet object containing user credentials.
        model: model to be pushed.
        repo_name: the name of huggingface repo
        uid: uid of subnet 9
    Returns:
        None.
    """
    model.push_to_hub(repo_name=repo_name, repo_id=f"subnet9_uid{uid}", token=token, safe_serialization=True)

# def get_run_id_from_huggingface(uid):
#     """
#     This function used to get hub_model_id from uid
#     Parameters:
#         model: uid
#         *
#         *
#         *
#     Returns:
#        hub_model_id: repo_name/repo_id
#     """

#     return hub_model_id

def model_size_valid(model):
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
    if model_size > 122200000 or model_size <82000000:
        warnings.warn("This model size is not Valid, please make sure you model parameter size is between 82M and 122M .")
        return False
    return True


def save( wallet, model ):
    """
    Saves the model state to your wallet path.

    Parameters:
        wallet: Wallet object containing user credentials.
        model: The model to be saved.

    Returns:
        None.
    """
    if model_size_valid(model):
        _model_path = model_path(wallet)
        if not os.path.exists(os.path.dirname(_model_path)):
            os.makedirs(os.path.dirname(_model_path), exist_ok=True)
        # Save the model state to the specified path
        save_model( model, _model_path )
    else:
        raise ValueError('Model failed to save.')



def load_from_hf(wallet, repo_name, token, uid, device: str = 'cpu'):
    """
    Loads the model state to your wallet path.

    Parameters:
        wallet: Wallet object containing user credentials.

    Returns:
        model: model loaded under wallet path.
    """
    
    model = AutoModelForCausalLM.from_pretrained(f"{repo_name}/subnet9_uid{uid}", use_safetensors=True) 
    if model_size_valid(model):
        save(wallet, model)
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
    
    _model_path = model_path(wallet)
    model = AutoModelForCausalLM.from_pretrained(_model_path) 
    if model_size_valid(model):
        return model
    else:
        raise ValueError('Model failed to load.')

def update( wallet, model , repo_name, token, uid):
    _run = init( wallet )
    save( wallet, model )
    push( model, repo_name, token, uid)
    _run.finish()


