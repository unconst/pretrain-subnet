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
import copy
import time
import json
import torch
import wandb
import typing
import pretrain
import bittensor as bt
from datetime import datetime


def check_run_exists( uid, metadata: dict, metagraph ):
    try:
        expected_runid = metadata['runid']
        api = wandb.Api( timeout = 100 )
        run = api.run(f"opentensor-dev/{pretrain.WANDB_PROJECT}/{expected_runid}")
        assert run.config['uid'] == uid
        assert run.config['hotkey'] == metagraph.hotkeys[uid]
        return True
    except Exception as e:
        bt.logging.debug(f'Check run failed with error: {e}')
        return False

def check_model_run_validity( run: 'wandb.run', metagraph: typing.Optional[ bt.metagraph ] = None  ) -> typing.Tuple[bool, str]:
    """
        Checks the validity of a Weights & Biases (wandb) run against a metagraph.

        This function verifies the integrity and authenticity of a wandb run by checking its hotkey and signature 
        against the provided metagraph. It also validates the presence of a model artifact file ('model.pth') 
        and its timestamp.

    Parameters:
        run (wandb.run): The wandb run to be validated.
        metagraph: The metagraph against which the run's validity is checked.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating the validity of the run and a string message 
                      explaining the validity status.

    Note:
        - The function requires the hotkey and signature to be present in the run's configuration.
        - It assumes the presence of 'model.pth' as a key artifact in the run.
        - The function catches and handles exceptions, providing an appropriate message in case of failure.
    """
    if not metagraph: metagraph = bt.subtensor().metagraph( pretrain.NETUID )
    try:
        # Extract hotkey and signature from the run's configuration
        hotkey = run.config['hotkey']
        signature = run.config['signature']

        # Check if the hotkey is registered in the metagraph
        if hotkey not in metagraph.hotkeys:
            return False, f'Failed Signature: The hotkey: {hotkey} is not in the metagraph.'

        # Verify the signature using the hotkey
        if not bt.Keypair(ss58_address=hotkey).verify(run.id, bytes.fromhex(signature)):
            return False, f'Failed Signature: The signature: {signature} is not valid'

        # Attempt to access the model artifact file
        try: 
            model_artifact = run.file('model.pth')
        except: 
            return False, f'Failed Signature: Does not have a model.pth file'

        # Check and convert the updated at timestamp
        try: 
            int(datetime.strptime(model_artifact.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
        except: 
            return False, f'Failed validity: Does not have a valid model.pth file'

        # The run has a valid signature
        return True, f'Passed Validity check.'
    except Exception as e:
        # Handle exceptions during the validity check
        return False, f'Failed Signature: An exception occurred while checking the signature with error: {e}'


def check_delta_run_validity( run: 'wandb.run', metagraph: typing.Optional[ bt.metagraph ] = None  ) -> typing.Tuple[bool, str]:
    """
        Checks the validity of a Weights & Biases (wandb) delta run against a metagraph.

        This function verifies the integrity and authenticity of a wandb run by checking its hotkey and signature 
        against the provided metagraph. It also validates the presence of a delta artifact file ('delta.pth') 
        and its timestamp.

    Parameters:
        run (wandb.run): The wandb run to be validated.
        metagraph: The metagraph against which the run's validity is checked.

    Returns:
        Tuple[bool, str]: A tuple containing a boolean indicating the validity of the run and a string message 
                      explaining the validity status.

    Note:
        - The function requires the hotkey and signature to be present in the run's configuration.
        - It assumes the presence of 'delta.pth' as a key artifact in the run.
        - The function catches and handles exceptions, providing an appropriate message in case of failure.
    """
    if not metagraph: metagraph = bt.subtensor().metagraph( pretrain.NETUID )
    try:
        # Extract hotkey and signature from the run's configuration
        hotkey = run.config['hotkey']
        signature = run.config['signature']

        # Check if the hotkey is registered in the metagraph
        if hotkey not in metagraph.hotkeys:
            return False, f'Failed Signature: The hotkey: {hotkey} is not in the metagraph.'

        # Verify the signature using the hotkey
        if not bt.Keypair(ss58_address=hotkey).verify(run.id, bytes.fromhex(signature)):
            return False, f'Failed Signature: The signature: {signature} is not valid'

        # Attempt to access the delta artifact file
        try: 
            delta_artifact = run.file('delta.pth')
        except: 
            return False, f'Failed Signature: Does not have a delta.pth file'

        # Check and convert the updated at timestamp
        try: 
            int(datetime.strptime(delta_artifact.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
        except: 
            return False, f'Failed validity: Does not have a valid delta.pth file'

        # The run has a valid signature
        return True, f'Passed Validity check.'
    except Exception as e:
        # Handle exceptions during the validity check
        return False, f'Failed Signature: An exception occurred while checking the signature with error: {e}'


def update_model_on_run(model: torch.nn.Module, run: 'wandb.run', path: str = os.path.expanduser('~/tmp/model.pth')):
    """
    Saves the state of a given model and updates the corresponding Weights & Biases (wandb) run with the model file.

    This function serializes the state of the provided PyTorch model and saves it to a specified file path. 
    It then uses the wandb API to log this file to the associated run, allowing for tracking and versioning of model states in wandb.

    Parameters:
        model (torch.nn.Module): The PyTorch model whose state is to be saved.
        run (wandb.run): The wandb run to which the model state will be logged.
        path (str): The file path where the model state will be saved. Defaults to '~/tmp/model.pth'.

    Note:
        - The function does not perform any checks on the validity of the provided model or wandb run.
        - The default save path is in the user's home directory under 'tmp', which should be verified or changed based on the system setup.
    """
    # Create the directory for the model path if it does not exist
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save the model state to the specified path
    torch.save(model.state_dict(), path)
    # Log the saved model file to wandb run.
    run.save( path )

def update_delta_on_run(delta: torch.nn.Module, run: 'wandb.run', path: str = os.path.expanduser('~/tmp/delta.pth')):
    """
    Saves the state of a given delta and updates the corresponding Weights & Biases (wandb) run with the delta file.

    This function serializes the state of the provided PyTorch delta and saves it to a specified file path. 
    It then uses the wandb API to log this file to the associated run, allowing for tracking and versioning of model states in wandb.

    Parameters:
        delta (torch.nn.Module): The PyTorch delta whose state is to be saved.
        run (wandb.run): The wandb run to which the delta state will be logged.
        path (str): The file path where the delta state will be saved. Defaults to '~/tmp/delta.pth'.

    Note:
        - The function does not perform any checks on the validity of the provided delta or wandb run.
        - The default save path is in the user's home directory under 'tmp', which should be verified or changed based on the system setup.
    """
    # Create the directory for the model path if it does not exist
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    # Save the model state to the specified path
    torch.save( delta.state_dict(), path )
    # Log the saved model file to wandb run.
    run.save( path )

def get_run_for_uid( uid: int, metagraph: typing.Optional[ bt.metagraph ] = None ) -> typing.Optional['wandb.run']:
    """
    Retrieves a specific Weights & Biases (wandb) run for a given user ID (uid).

    This function queries the wandb API for runs associated with a specific user ID.
    It filters runs based on the version, type, run name pattern, and hotkey corresponding to the UID.
    The function returns the most recent valid run for the given UID.

    Parameters:
        uid (int): The user ID for which the wandb run is to be retrieved.

    Returns:
        Optional[wandb.run]: The most recent valid wandb run for the given UID, or None if no valid run is found.

    Note:
    - This function assumes that the metagraph and wandb API settings are correctly configured.
    - It relies on specific configuration fields (version, type, run_name, hotkey) being present in the wandb run's metadata.
    """
    if not metagraph: metagraph = bt.subtensor().metagraph( pretrain.NETUID )
    api = wandb.Api(timeout=100)
    expected_hotkey = metagraph.hotkeys[uid]

    # Retrieve runs from the wandb project for this uid
    runs = api.runs(
        f"opentensor-dev/{pretrain.WANDB_PROJECT}",
        filters={
            "config.version": pretrain.__version__,
            "config.type": "miner",
            "config.run_name": {
                "$regex": f"miner-{uid}-.*"
            },
            "config.hotkey": expected_hotkey,
        }
    )

    # Iterate through runs. Newer runs are processed first.
    for run in runs:
        check_model_run_validity(run, metagraph)
        return run
    
    return None

def load_model_metadata_for_uid( uid: int ):
    """
        Retrieves the file path to the model checkpoint for uid with associated timestamps and verion.
    Args:
        uid: Uid to find metadata on.    
    """
    try:
        # Fill metadata from files and check if we can load weights.
        model_dir = os.path.join(pretrain.netuid_dir, 'models', str(uid))
        try:
            with open(os.path.join(model_dir, 'metadata.json'), 'r') as f: 
                meta = json.load(f)
        except Exception as e:
            # bt.logging.trace(f'Failed Metadata: uid:{uid}, no file under path:{model_dir}')
            return None
        if 'version' not in meta or 'timestamp' not in meta or 'runid' not in meta:
            bt.logging.trace(f'Failed Metadata: uid:{uid}, metadata file corrupted.')
            return None
        else:
            # Valid metadata.
            return meta
    except Exception as e:
        # Skip this UID if any error occurs during loading of model or timestamp.
        bt.logging.trace(f'Failed Metadata: uid:{uid}, metadata could not be loaded with error:{e}')
        return None

def update_model_for_uid(uid: int, metagraph: typing.Optional[bt.metagraph] = None) -> bool:
    """
    Updates the model for a given user ID (uid) if there is a newer valid run.

    This function checks for the latest valid run for the specified uid and updates the local model files 
    if there is a newer version available. It manages the metadata and model files based on the latest run's 
    information. If no valid run is found, it cleans up any existing model and metadata files.

    Parameters:
        uid (int): The user ID for which the model update is to be checked and performed.
        metagraph (Optional[bt.metagraph]): The metagraph to use for finding the latest valid run. 
                                        If not provided, it's fetched using pretrain.NETUID.

    Returns:
        bool: True if the model was updated, False otherwise.

    Note:
        - The function assumes the existence of specific directories and file paths as defined in 'pretrain'.
        - It manages model files in a directory structure based on the uid.
    """
    # Get the latest valid run for this uid
    if not metagraph: 
        metagraph = bt.subtensor().metagraph(pretrain.NETUID)

    latest_valid_run = get_run_for_uid(uid, metagraph=metagraph)
    expected_hotkey = metagraph.hotkeys[uid]

    # Paths for model and metadata
    models_dir = os.path.join(pretrain.netuid_dir, 'models', str(uid))
    metadata_file = os.path.join(models_dir, 'metadata.json')
    model_path = os.path.join(models_dir, 'model.pth')

    # If no valid runs, delete existing model and metadata
    if latest_valid_run is None:
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        if os.path.exists(model_path):
            os.remove(model_path)
            bt.logging.debug(f'Deleting {uid} model with no run.')
        return False

    # Create model directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)

    # Load model artifact and get timestamp
    model_artifact = latest_valid_run.file('model.pth')
    timestamp = int(datetime.strptime(model_artifact.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
    current_meta = load_model_metadata_for_uid(uid)

    # Update metadata file
    with open(metadata_file, 'w') as f: 
        json.dump({
            'timestamp': timestamp, 
            'runid': latest_valid_run.id,
            'model_path': model_path,
            'version': latest_valid_run.config['version'],
            'blacklisted': False,
            'last_update': time.time(),
            'uid': uid,
            'hotkey': expected_hotkey,
        }, f)

    # Check if model needs updating
    if current_meta is not None and current_meta.get('timestamp', -1) == timestamp:
        bt.logging.trace(f'Model is up to date: uid: {uid}, under path: {models_dir}, with timestamp: {timestamp}')
        return False
    else:
        # Download the updated model artifact
        model_artifact.download(replace=True, root=models_dir)
        bt.logging.success(f'Updated model: uid: {uid}, under path: {models_dir}, with timestamp: {timestamp}')
        return True


def update_delta_for_uid( uid:int, metagraph: typing.Optional[ bt.metagraph ] = None ):

    """
    Updates the delta for a given user ID (uid) if there is a newer valid run.

    This function checks for the latest valid run for the specified uid and updates the local delta files 
    if there is a newer version available. It manages the metadata and delta files based on the latest run's 
    information. If no valid run is found, it cleans up any existing delta and metadata files.

    Parameters:
        uid (int): The user ID for which the delta update is to be checked and performed.
        metagraph (Optional[bt.metagraph]): The metagraph to use for finding the latest valid run. 
                                        If not provided, it's fetched using pretrain.NETUID.

    Returns:
        bool: True if the delta was updated, False otherwise.

    Note:
        - The function assumes the existence of specific directories and file paths as defined in 'pretrain'.
        - It manages delta files in a directory structure based on the uid.
    """
    # Get the latest valid run for this uid
    if not metagraph: 
        metagraph = bt.subtensor().metagraph(pretrain.NETUID)

    latest_valid_run = get_run_for_uid(uid, metagraph=metagraph)
    expected_hotkey = metagraph.hotkeys[uid]

    # Paths for delta and metadata
    delta_dir = os.path.join(pretrain.netuid_dir, 'models', str(uid))
    dela_metadata_file = os.path.join(delta_dir, 'metadata.json')
    delta_path = os.path.join(delta_dir, 'delta.pth')

    # If no valid runs, delete existing model and metadata
    if latest_valid_run is None:
        if os.path.exists(dela_metadata_file):
            os.remove(dela_metadata_file)
        if os.path.exists(delta_path):
            os.remove(delta_path)
            bt.logging.debug(f'Deleting {uid} model with no run.')
        return False

    # Create model directory if it doesn't exist
    os.makedirs(delta_path, exist_ok=True)

    # Load model artifact and get timestamp
    model_artifact = latest_valid_run.file('model.pth')
    timestamp = int(datetime.strptime(model_artifact.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
    current_meta = load_delta_metadata_for_uid(uid)

    # Update metadata file
    with open(dela_metadata_file, 'w') as f: 
        json.dump({
            'timestamp': timestamp, 
            'runid': latest_valid_run.id,
            'delta_path': delta_path,
            'version': latest_valid_run.config['version'],
            'blacklisted': False,
            'last_update': time.time(),
            'uid': uid,
            'hotkey': expected_hotkey,
        }, f)

    # Check if model needs updating
    if current_meta is not None and current_meta.get('timestamp', -1) == timestamp:
        bt.logging.trace(f'Model is up to date: uid: {uid}, under path: {delta_dir}, with timestamp: {timestamp}')
        return False
    else:
        # Download the updated model artifact
        model_artifact.download(replace=True, root=delta_dir)
        bt.logging.success(f'Updated model: uid: {uid}, under path: {delta_dir}, with timestamp: {timestamp}')
        return True


def load_delta_metadata_for_uid( uid: int ):
    """
        Retrieves the file path to the model checkpoint for uid with associated timestamps and verion.
    Args:
        uid: Uid to find metadata on.    
    """
    try:
        # Fill metadata from files and check if we can load weights.
        model_dir = os.path.join(pretrain.netuid_dir, 'models', str(uid))
        try:
            with open(os.path.join(model_dir, 'delta_metadata.json'), 'r') as f: 
                meta = json.load(f)
        except Exception as e:
            # bt.logging.trace(f'Failed Metadata: uid:{uid}, no file under path:{model_dir}')
            return None
        if 'version' not in meta or 'timestamp' not in meta or 'runid' not in meta:
            bt.logging.trace(f'Failed Metadata: uid:{uid}, metadata file corrupted.')
            return None
        else:
            # Valid metadata.
            return meta
    except Exception as e:
        # Skip this UID if any error occurs during loading of model or timestamp.
        bt.logging.trace(f'Failed Metadata: uid:{uid}, metadata could not be loaded with error:{e}')
        return None
      

def apply_delta( base_model: torch.nn.Module, delta_model: torch.nn.Module, device: str = 'cpu' ) -> torch.nn.Module:
    """
    Applies the changes from a delta model onto a base model.

    This function creates a copy of the base model and then iterates through its parameters, 
    updating them with the corresponding parameters from the delta model.

    Parameters:
        base_model (torch.nn.Module): The original model to which the delta will be applied.
        delta_model (torch.nn.Module): The delta model containing the updates.
        device (str): The device (e.g., 'cpu' or 'gpu') to move the models for processing. Defaults to 'cpu'.

    Returns:
        torch.nn.Module: The updated model with the delta applied.
    """

    # Clone the base model to avoid modifying it directly
    new_model = copy.deepcopy(base_model)
    # Move the new model to the specified device
    new_model.to(device)

    # Iterate over the parameters of the base model and the delta model
    for (base_param_name, base_param), (delta_param_name, delta_param) in zip(base_model.named_parameters(), delta_model.named_parameters()):
        # Ensure the parameter names in both models match
        if base_param_name == delta_param_name:
            # Move the delta parameter to the same device as the new model
            delta_param = delta_param.to(device)
            # Apply the delta and update the parameter in the new model
            # Using in-place addition to be memory efficient
            new_model.state_dict()[base_param_name].add_(delta_param.data)

    return new_model

def get_and_update_delta_for_uid(uid: int, device: str = 'cpu') -> typing.Optional[torch.nn.Module]:
    """
    Retrieves and updates the delta (a form of model update) for a specific user ID (uid).
    This function tries to update the delta weights and then loads them into the model.

    Parameters:
    uid (int): The user ID for which the delta is to be retrieved and updated.
    device (str): The device on which the model is to be loaded. Defaults to 'cpu'.

    Returns:
    torch.nn.Module or None: The updated model if successful, None otherwise.
    """
    try:
        update_delta_for_uid(uid)
        delta = pretrain.model.get_model()
        delta_meta = load_delta_metadata_for_uid(uid)
        delta_weights = torch.load(delta_meta['delta_path'], map_location=torch.device(device))
        delta.load_state_dict(delta_weights)
        return delta
    except Exception as e:
        bt.logging.debug(f'Error loading delta for uid: {uid} with error: {e}')
        return None

def get_delta_for_uid(uid: int, device: str = 'cpu') -> typing.Optional[torch.nn.Module]:
    """
    Retrieves the delta for a specific user ID (uid) without updating it.
    This function loads the delta weights into the model.

    Parameters:
    uid (int): The user ID for which the delta is to be retrieved.
    device (str): The device on which the model is to be loaded. Defaults to 'cpu'.

    Returns:
    torch.nn.Module or None: The model with loaded delta if successful, None otherwise.
    """
    try:
        delta = pretrain.model.get_model()
        delta_meta = load_delta_metadata_for_uid(uid)
        delta_weights = torch.load(delta_meta['delta_path'], map_location=torch.device(device))
        delta.load_state_dict(delta_weights)
        return delta
    except Exception as e:
        bt.logging.debug(f'Error loading delta for uid: {uid} with error: {e}')
        return None

def get_and_update_model_for_uid(uid: int, device: str = 'cpu') -> typing.Optional[torch.nn.Module]:
    """
    Retrieves and updates the model for a specific user ID (uid).
    This function tries to update the model weights and then loads them into the model.

    Parameters:
    uid (int): The user ID for which the model is to be retrieved and updated.
    device (str): The device on which the model is to be loaded. Defaults to 'cpu'.

    Returns:
    torch.nn.Module or None: The updated model if successful, None otherwise.
    """
    try:
        update_model_for_uid(uid)
        model = pretrain.model.get_model()
        model_meta = load_model_metadata_for_uid(uid)
        model_weights = torch.load(model_meta['model_path'], map_location=torch.device(device))
        model.load_state_dict(model_weights)
        return model
    except Exception as e:
        bt.logging.debug(f'Error loading model for uid: {uid} with error: {e}')
        return None

def get_model_for_uid(uid: int, device: str = 'cpu') -> typing.Optional[torch.nn.Module]:
    """
    Retrieves the model for a specific user ID (uid) without updating it.
    This function loads the model weights into the model.

    Parameters:
    uid (int): The user ID for which the model is to be retrieved.
    device (str): The device on which the model is to be loaded. Defaults to 'cpu'.

    Returns:
    torch.nn.Module or None: The model if successful, None otherwise.
    """
    try:
        model = pretrain.model.get_model()
        model_meta = load_metadata_for_uid(uid)
        model_weights = torch.load(model_meta['model_path'], map_location=torch.device(device))
        model.load_state_dict(model_weights)
        return model
    except Exception as e:
        bt.logging.debug(f'Error loading model for uid: {uid} with error: {e}')
        return None

