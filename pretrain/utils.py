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

UPDATE_TIMEOUT: int = 60 * 60 * 6

def check_run_validity( run, metagraph ) -> typing.Tuple[ bool, str ]:
    try:
        hotkey = run.config['hotkey']
        signature = run.config['signature']
        if hotkey not in metagraph.hotkeys:
            # The hotkey is not registered.
            return False, f'Failed Signature: The hotkey: {hotkey} is not in the metagraph.'
        if not bt.Keypair(ss58_address=hotkey).verify(run.id, bytes.fromhex(signature)): 
            # The signature is invalid
            return False, f'Failed Signature: The signature: {signature} is not valid'
        # Attempt to access the model artifact file.
        try: 
            model_artifact = run.file('model.pth')
        except: 
            return False, f'Failed Signature: Does not have a model.pth file'
        # Check convert the updated at timestamp.
        try: 
            int(datetime.strptime(model_artifact.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
        except: 
            return False, f'Failed validity: Does not have a model.pth file'
        # The run has a valid signature. 
        return True, f'Passed Validity check.'
    except Exception as e:
        # An exception occurec when checking the signature.
        return False, f'Failed Signature: An exception occured while checking the signature with error: {e}'

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

def update_model_for_uid( uid:int, metagraph: typing.Optional[ bt.metagraph ] = None ):

    if not metagraph: metagraph = bt.subtensor().metagraph( pretrain.NETUID )
    # Retrieve runs from the wandb project for this uid
    api = wandb.Api( timeout = 100 )
    expected_hotkey = metagraph.hotkeys[uid]
    runs = api.runs(
        f"opentensor-dev/{pretrain.WANDB_PROJECT}",
        filters={
            "config.version": pretrain.__version__,
            "config.type": "miner",
            "config.run_name": {
                "$regex": f"miner-{uid}-.*"
            },
            "config.hotkey": expected_hotkey,
        },
        # per_page = 3,
    )
    models_dir = os.path.join( pretrain.netuid_dir, 'models', str(uid) )
    metadata_file = os.path.join( models_dir, 'metadata.json' )
    model_path = os.path.join( models_dir, 'model.pth' )

    # Delete models where there is no file.
    if len( runs ) == 0:
        if os.path.exists(metadata_file):
            os.remove(metadata_file)
        if os.path.exists(model_path):
            os.remove(model_path)
            bt.logging.debug(f'Deleting {uid} model with no run.')
        return False

    # Iterate through runs. Newer runs first.
    for run in runs:
        bt.logging.trace(f'check run: {run.id}')

        # Check if the run is valid.
        valid, reason = check_run_validity( run, metagraph )
        if valid:
            bt.logging.trace(f'Run:{run.id}, for uid:{uid} was valid.')
            
            # Run artifact.
            try:
                model_artifact = run.file('model.pth')
            except:
                # No model, continue.
                continue

            # Define the local model directory and timestamp file paths
            timestamp = int(datetime.strptime(model_artifact.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
            current_meta = load_metadata_for_uid( uid )  
                      
            # The run is valid, lets update it
            os.makedirs( models_dir , exist_ok=True)
            with open(metadata_file, 'w') as f: 
                json.dump( 
                    { 
                        'timestamp': timestamp, 
                        'runid': run.id,
                        'model_path': model_path,
                        'version': run.config['version'],
                        'blacklisted': False,
                        'last_update': time.time(),
                        'uid': uid,
                        'hotkey': expected_hotkey,
                    }, f)

            # Check to see if model needs updating.
            if current_meta != None and current_meta.get('timestamp', -1) == timestamp:
                bt.logging.trace( f'Model is up to date: uid: {uid}, under path: {models_dir}, with timestamp: { timestamp }')
                return False
            else:
                model_artifact.download( replace=True, root=models_dir)
            bt.logging.success( f'Updated model: uid: {uid}, under path: {models_dir}, with timestamp: { timestamp }')
            return True

        else:
            # The run failed the signature check. Moving to the next run.
            bt.logging.trace(f'Run:{run.id}, for uid:{uid} was not valid with error: {reason}')
            continue

    # Deleting model path if no valid runs.
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    if os.path.exists(model_path):
        os.remove(model_path)
        bt.logging.debug(f'Deleting {uid} model with no valid run.')
    return False 

def load_metadata_for_uid( uid: int ):
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

def update_delta_for_uid( uid:int, metagraph: typing.Optional[ bt.metagraph ] = None ):

    if not metagraph: metagraph = bt.subtensor().metagraph( pretrain.NETUID )
    # Retrieve runs from the wandb project for this uid
    api = wandb.Api( timeout = 100 )
    expected_hotkey = metagraph.hotkeys[uid]
    runs = api.runs(
        f"opentensor-dev/{pretrain.WANDB_PROJECT}",
        filters={
            "config.version": pretrain.__version__,
            "config.type": "delta-miner",
            "config.run_name": {
                "$regex": f"delta-miner-{uid}-.*"
            },
            "config.hotkey": expected_hotkey,
        },
    )
    models_dir = os.path.join( pretrain.netuid_dir, 'models', str(uid) )
    delta_metadata_file = os.path.join( models_dir, 'delta_metadata.json' )
    delta_path = os.path.join( models_dir, 'delta.pth' )

    # Delete models where there is no file.
    if len( runs ) == 0:
        if os.path.exists(delta_metadata_file):
            os.remove(delta_metadata_file)
        if os.path.exists(delta_path):
            os.remove(delta_path)
            bt.logging.debug(f'Deleting {uid} model with no run.')
        return False

    # Iterate through runs. Newer runs first.
    for run in runs:
        bt.logging.trace(f'check run: {run.id}')

        # Check if the run is valid.
        valid, reason = check_run_validity( run, metagraph )
        if valid:
            bt.logging.trace(f'Run:{run.id}, for uid:{uid} was valid.')
            
            # Run artifact.
            try:
                delta_artifact = run.file('delta.pth')
            except:
                # No model, continue.
                continue

            # Define the local model directory and timestamp file paths
            timestamp = int(datetime.strptime(delta_artifact.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
            current_meta = load_metadata_for_uid( uid )  
                      
            # The run is valid, lets update it
            os.makedirs( models_dir , exist_ok=True)
            with open(delta_metadata_file, 'w') as f: 
                json.dump( 
                    { 
                        'timestamp': timestamp, 
                        'runid': run.id,
                        'delta_path': delta_path,
                        'version': run.config['version'],
                        'blacklisted': False,
                        'last_update': time.time(),
                        'uid': uid,
                        'hotkey': expected_hotkey,
                    }, f)

            # Check to see if model needs updating.
            if current_meta != None and current_meta.get('timestamp', -1) == timestamp:
                bt.logging.trace( f'Model is up to date: uid: {uid}, under path: {models_dir}, with timestamp: { timestamp }')
                return False
            else:
                delta_artifact.download( replace=True, root=models_dir )
            bt.logging.success( f'Updated model: uid: {uid}, under path: {models_dir}, with timestamp: { timestamp }')
            return True

        else:
            # The run failed the signature check. Moving to the next run.
            bt.logging.trace(f'Run:{run.id}, for uid:{uid} was not valid with error: {reason}')
            continue

    # Deleting model path if no valid runs.
    if os.path.exists(delta_metadata_file):
        os.remove(delta_metadata_file)
    if os.path.exists(delta_path):
        os.remove(delta_path)
        bt.logging.debug(f'Deleting {uid} delta with no valid run.')
    return False

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
        model_meta = load_metadata_for_uid(uid)
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

