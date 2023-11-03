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

# pip install huggingface_hub
# install git-lfs
# huggingface-cli login with personal access token

import os
import time
import torch
import string
import random
import argparse
import pretrain
import traceback
import bittensor as bt
from huggingface_hub import HfApi, HfFolder, Repository
import subprocess


# === Config ===
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model_path", type = str, help="Run name.", default='~/model.pth' )
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)
    return config

config = get_config()

# === Bittensor objects ===
bt.logging( config = config )
bt.logging.success( config )
wallet = bt.wallet( config = config ) 
subtensor = bt.subtensor( config = config )
metagraph = subtensor.metagraph( pretrain.NETUID )
if wallet.hotkey.ss58_address not in metagraph.hotkeys: raise Exception("You are not registered. Use `btcli s recycle_register` to register.")
my_uid = metagraph.hotkeys.index( wallet.hotkey.ss58_address )
bt.logging.success( f'You are registered with address: {wallet.hotkey.ss58_address} and uid: {my_uid}' )

# === Authenticate to Hugging Face ===
api = HfApi()
user = api.whoami(HfFolder.get_token())
username = user['name']

# === Prepare Hugging Face repository ===
model_name = f"{wallet.hotkey.ss58_address}"
repo_name = f"{username}/{model_name}"
repo_url = api.create_repo(repo_name, private=False, exist_ok=True)
repo_local_path = os.path.expanduser(f"~/pretrain-subnet/neurons/pretraining_model/{model_name}")

repo = Repository(local_dir=repo_local_path, clone_from=repo_url)
bt.logging.info(f"Cloned repository {repo_name} to {repo_local_path}")

# === Function to get file modification timestamps ===
def get_all_file_paths_and_timestamps(directory):
    file_paths_and_timestamps = {}
    for subdir, dirs, files in os.walk(directory):
        if '.git' in subdir:
            continue 
        for file in files:
            file_path = os.path.join(subdir, file)
            file_paths_and_timestamps[file_path] = os.path.getmtime(file_path)
    return file_paths_and_timestamps

# Initialize the dictionary to store the file modification times
file_mod_times = get_all_file_paths_and_timestamps(repo_local_path)

def give_repo( synapse: pretrain.protocol.GetRepo ) -> pretrain.protocol.GetRepo:
    synapse.huggingface_repo = repo_name
    return synapse

def run_command(command, cwd):
    result = subprocess.run(command, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise subprocess.CalledProcessError(result.returncode, result.args, output=result.stdout, stderr=result.stderr)
    return result.stdout.strip()

run_command(['git', 'lfs', 'track', "*.safetensors"], cwd=repo_local_path)
run_command(['git', 'lfs', 'track', "*.bin"], cwd=repo_local_path)
run_command(['git', 'lfs', 'track', "*.msgpack"], cwd=repo_local_path)
run_command(['git', 'lfs', 'track', "*.ot"], cwd=repo_local_path)

# Initialize the dictionary to store the file modification times
file_mod_times = get_all_file_paths_and_timestamps(repo_local_path)

# === Axon ===
axon = bt.axon( 
    wallet = wallet, 
    config = config 
).attach( 
    forward_fn = give_repo,
).start()
bt.logging.success( f'Started axon.' )

axon.start().serve( 
    subtensor = subtensor,
    netuid = pretrain.NETUID,
)
bt.logging.success( f'Served Axon.' )

# === Run ===
step = 1
subtensor.set_weights (
    netuid = pretrain.NETUID,
    wallet = wallet, 
    uids = [my_uid], 
    weights = [1.0], 
    wait_for_inclusion=False,
)
while True:
    try:
        bt.logging.info(f"checking for updates...")
        
        current_file_mod_times = get_all_file_paths_and_timestamps(repo_local_path)
        files_changed = False

        # Check for new or updated files
        for file_path, current_mod_time in current_file_mod_times.items():
            if file_path not in file_mod_times or current_mod_time != file_mod_times[file_path]:
                if '.git' not in file_path:  # Ignore changes in the .git directory
                    bt.logging.info(f"Detected change in file: {file_path}")
                    files_changed = True
                    file_mod_times[file_path] = current_mod_time
                    run_command(['git', 'add', file_path], cwd=repo_local_path) 

        # Check for deleted files
        deleted_files = [file for file in file_mod_times if file not in current_file_mod_times]
        if deleted_files:
            files_changed = True
            for file_path in deleted_files:
                if '.git' not in file_path:  # Ignore changes in the .git directory
                    bt.logging.info(f"Detected deletion of file: {file_path}")
                    try:
                        run_command(['git', 'rm', file_path], cwd=repo_local_path)
                    except subprocess.CalledProcessError as e:
                        bt.logging.error(f"Failed to remove {file_path}: {e.stderr}")
                    del file_mod_times[file_path]  # Remove from our tracked files

        run_command(['git', 'add', '.gitattributes'], cwd=repo_local_path) 

        # If any file was changed, push to Hugging Face
        if files_changed:
            bt.logging.info("Pushing updates to Hugging Face Hub")
            # No need to iterate over files again, as we've already handled them
            try:
                repo.git_commit("Update model files")
                repo.git_push()
                bt.logging.info("Pushed updated model to Hugging Face Hub")
            except EnvironmentError as e:
                if 'nothing to commit' in str(e):
                    bt.logging.info("No changes to commit.")
                else:
                    raise  # Re-raise the exception if it's not a 'nothing to commit' case

        time.sleep( 10 )
        step += 1

        if step % 100 == 0:
            subtensor.set_weights (
                netuid = pretrain.NETUID,
                wallet = wallet, 
                uids = [my_uid], 
                weights = [1.0], 
                wait_for_inclusion=False,
            )

    except Exception as e:
        bt.logging.error(f"error in main block {traceback.format_exc()}" )
        continue




