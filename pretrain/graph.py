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
import time
import json
import torch
import typing
import pretrain
import shutil
import pretrain.mining as mining
import bittensor as bt
from datetime import datetime
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM

def best_uid( metagraph: typing.Optional[ bt.metagraph ] = None ):
    if not metagraph: metagraph = bt.subtensor().metagraph(pretrain.NETUID)
    return max( range(256), key=lambda uid: metagraph.I[uid].item() )

def best_model( subtensor, metagraph: typing.Optional[ bt.metagraph ] = None, device:str = 'cpu '):
    if not metagraph: metagraph = bt.subtensor().metagraph(pretrain.NETUID)
    _best_uid = best_uid( metagraph )
    return model( subtensor, _best_uid, device = device )

def timestamp( uid: int ) -> typing.Optional[ int ]:
    try:
        return metadata(uid)['timestamp']
    except Exception as e:
        bt.logging.debug('Failed to get timestamp for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} )'.format(uid, uid))
        return None
    

def version( uid: int ) -> typing.Optional[ str ]:
    try:
        return metadata(uid)['version']
    except Exception as e:
        bt.logging.debug('Failed to get version for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} )'.format(uid, uid))
        return None

def last_download( uid: int ) -> typing.Optional[ float ]:
    try:
        _metadata = metadata(uid)
        if 'last_download' not in _metadata:
            return _metadata[ 'last_update' ]
        else:
            return _metadata[ 'last_download' ]
    except Exception as e:
        bt.logging.debug('Failed to get last_download for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} ) with error: { }'.format(uid, uid, e))
        return None

def hotkey( uid: int ) -> typing.Optional[ str ]:
    try:
        return metadata(uid)['hotkey']
    except Exception as e:
        bt.logging.debug('Failed to get hotkey for uid: {}, try pulling data for this uid with pretrain.graph.sync( {} )'.format(uid, uid))
        return None


def metadata( uid: int ):
    """
        Retrieves the file path to the model checkpoint for uid with associated timestamps.
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
            bt.logging.trace(f'Failed Metadata: uid:{uid}, no file under path:{model_dir}')
            return None
        if 'timestamp' not in meta or 'uid' not in meta:
            bt.logging.debug(f'Failed Metadata: uid:{uid}, metadata file corrupted.')
            return None
        else:
            # Valid metadata.
            return meta
    except Exception as e:
        # Skip this UID if any error occurs during loading of model or timestamp.
        bt.logging.debug(f'Failed Metadata: uid:{uid}, metadata could not be loaded with error:{e}')
        return None



def sync( subtensor, uid: int, metagraph: typing.Optional[bt.metagraph] = None ) -> bool:
    """
    Updates the model for a given user ID (uid) if there is a newer valid run.

    This function checks for the latest valid run for the specified uid and updates the local model files 
    if there is a newer version available. It manages the metadata and model files based on the latest run's 
    information. If no valid run is found, it cleans up any existing model and metadata files.

    Parameters:      
        subtensor: subtensor object, in here we will use subtensor block-chain infra to get time-stamps
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

    # Paths for model and metadata
    models_dir = os.path.join(pretrain.netuid_dir, 'models', str(uid))
    metadata_file = os.path.join(models_dir, 'metadata.json')
    expected_hotkey = metagraph.hotkeys[uid]
    try:
        commit = bt.extrinsics.serving.get_metadata( subtensor, netuid = pretrain.NETUID, hotkey=expected_hotkey)
        commitment = commit["info"]["fields"][0]
        repo_name = bytes.fromhex(commitment[list(commitment.keys())[0]][2:]).decode()
        # if repo not valid return 
        if not check_repo_valid(repo_name, uid):
            remove_local_files(metadata_file, models_dir)
            return False
    except:
        # if there is no valid record on chain, delete related files and return
        remove_local_files(metadata_file, models_dir)
        bt.logging.debug(f'Deleting {uid} model with no run.')
        return False
    
    # flag to determine if we need new model and update metadata
    changed_flag = False
    if os.path.exists(models_dir):
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            if metadata['hotkey']!=expected_hotkey or metadata['timestamp'] != commit['block']:
                changed_flag = True
                remove_local_files(metadata_file, models_dir)
    else:
        changed_flag=True            

    if changed_flag == False:
        metadata['last_update'] =  time.time()
        with open(metadata_file, 'w') as f: 
            json.dump(metadata, f)
        return False
    else:
        save_model( models_dir, repo_name)
        metadata = {
                'timestamp': commit['block'], 
                'model_path': repo_name,
                'blacklisted': False,
                'last_update': time.time(),
                'uid': uid,
                'hotkey': expected_hotkey,
        }
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f)
        return True

def save_model( models_dir, repo_name):
    """
    Saves the model state to your wallet path.

    Parameters:
        wallet: Wallet object containing user credentials.
        repo_name: The repo (huggingface) to be saved.

    Returns:
        model .
    """
    if not os.path.exists(os.path.dirname(models_dir)):
        os.makedirs(os.path.dirname(models_dir), exist_ok=True)
    try:
        # Save the model state to the specified path
        snapshot_download(repo_id=repo_name, \
                        local_dir=models_dir, \
                            local_dir_use_symlinks=False)
    except:
        shutil.rmtree(models_dir) 
        raise ValueError('Model failed to save.')

def remove_local_files(metadata_file, models_dir):
    """
    util function to remove invalid run's metadata and models.

    Parameters:
        metadata_file: path of metadata_file
        models_dir: path of models

    Returns:
        None .
    """
    if os.path.exists(metadata_file):
        os.remove(metadata_file)
    if os.path.exists(models_dir):
        shutil.rmtree(models_dir) 
        bt.logging.debug(f'No valid run, Deleting {metadata_file}  and {models_dir}')
def check_repo_valid(repo, uid):
    """
    Check if a HF repo is valid.

    Parameters:
        repo: The repo (huggingface) to be checked.
        uid (int): The user ID for which the model update is to be checked and performed.
    Returns:
        bool .
    """
    try:
        # todo replace with huggingface 
        model = AutoModelForCausalLM.from_pretrained(repo, use_safetensors=True) 
        if mining.model_size_valid(model):
            return True
        else:
            raise False
    except Exception as e:
        bt.logging.debug(f'Error loading model for uid: {uid} with error: {e}, try pulling data for this uid with pretrain.graph.pull( {uid} )')
        return False


def get_latest_model( subtensor, metadata):
    """
        Retrieves the uid info from subtensor
    Args:
        subtensor: subtensor object used to get commit info
        metadata: metadata of specific uid
    Returns:
        bool, Dict
    """
    try:
        commit = subtensor.get_commitment( netuid = pretrain.NETUID, uid = metadata['uid'] )
        key_name = list[commit['info']['fields'][0].keys()][0]
        repo_name = bytes.fromhex(commit['info']['fields'][0][key_name][2:]).decode()
        if metadata['timestamp'] != commit['block']:
            metadata['timestamp'] = commit['block']
            metadata['repo_name'] = repo_name
            return True, metadata
        return False, metadata
    except Exception as e:
        # Skip this UID if any error occurs during loading of model or timestamp.
        raise ValueError(f"Failed Metadata: uid:{metadata['uid']}, metadata could not be loaded with error:{e}")

def load_model(uid: int, device: str = 'cpu') -> typing.Optional[torch.nn.Module]:
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
        models_dir = os.path.join(pretrain.netuid_dir, 'models', str(uid))
        # todo replace with huggingface 
        model = AutoModelForCausalLM.from_pretrained(models_dir, use_safetensors=True, local_files_only=True) 
        if mining.model_size_valid(model):
            return model.to(device)
        else:
            raise ValueError('Model size not valid, skip.')
    except Exception as e:
        bt.logging.debug(f'Error loading model for uid: {uid} with error: {e}, try pulling data for this uid with pretrain.graph.pull( {uid} )')
        return None
    
