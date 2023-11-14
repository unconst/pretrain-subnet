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
import math
import time
import wandb
import torch
import typing
import string
import random
import argparse
import pretrain
import traceback
import threading
import bittensor as bt
from tqdm import tqdm
from typing import Dict, List
from rich.table import Table
from rich.console import Console

# Global artifact name
UPDATE_TIMEOUT = 60*60*2
ARTIFACT_NAME:str = "model.pth"

class Validator:

    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument( "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")
        parser.add_argument( '--wandb.off', dest = 'wandb.on', action='store_false', help='Turn off wandb logging.' )
        parser.add_argument( '--blocks_per_epoch', type=int, default=360, help='Number of blocks to wait before setting weights.' )
        parser.add_argument( '--pages_per_eval', type=int, default=3, help='Number of pages used to eval each step.' )
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)
        bt.wallet.add_args(parser)
        bt.axon.add_args(parser)
        config = bt.config(parser)
        config.full_path = os.path.expanduser(
            "{}/{}/{}/netuid{}/{}".format(
                config.logging.logging_dir,
                config.wallet.name,
                config.wallet.hotkey,
                pretrain.NETUID,
                "validator",
            )
        )
        if not os.path.exists(config.full_path):
            os.makedirs(config.full_path, exist_ok=True)
        return config

    def init_wandb(self):
        if self.config.wandb.on:
            run_name = f'validator-{self.uid}-' + ''.join(random.choice( string.ascii_uppercase + string.digits ) for i in range(10))
            self.config.uid = self.uid
            self.config.hotkey = self.wallet.hotkey.ss58_address
            self.config.run_name = run_name
            self.config.type = "validator"
            self.wandb_run =  wandb.init(
                name = run_name,
                anonymous = "allow",
                reinit = False,
                project = 'pretraining-subnet',
                entity = 'opentensor-dev',
                config = self.config,
                dir = self.config.full_path,
            )
            # Sign wandb run.
            wandb.init( project = pretrain.WANDB_PROJECT, entity="opentensor-dev" )
            self.config.signature = self.wallet.hotkey.sign( self.wandb_run.id.encode() ).hex()
            wandb.config.update( self.config, allow_val_change=True )

    def __init__(self ):
        self.config = Validator.config()
        bt.logging( config = self.config )

        # === Bittensor objects ====
        self.wallet = bt.wallet( config = self.config )
        self.subtensor = bt.subtensor( config = self.config )
        self.dendrite = bt.dendrite( wallet = self.wallet )
        self.metagraph = self.subtensor.metagraph( pretrain.NETUID )
        torch.backends.cudnn.benchmark = True
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys: raise Exception("You are not registered. Use `btcli s recycle_register` to register.")
        self.uid = self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )
        bt.logging.success( f'You are registered with address: {self.wallet.hotkey.ss58_address} and uid: {self.uid}' )
        self.init_wandb()

        # === Running args ===
        self.epoch_step = 0 
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()
        self.last_update_check = {}
        self.shouldeval = { uid: uid in self.metagraph.I.topk(10).indices.tolist() for uid in self.metagraph.uids.tolist()  }
        self.metadata = { uid: pretrain.utils.load_metadata_for_uid( uid ) for uid in self.metagraph.uids.tolist() }

        # == Initialize the update thread ==
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self.update_models, daemon=True)
        self.update_thread.start()

    def __del__(self):
        self.stop_event.set()
        self.update_thread.join()

    def update_models( self ):
        while not self.stop_event.is_set():
            # Go through sorted metadata, if the update interval has passed, update the model.
            self.metadata = { uid: pretrain.utils.load_metadata_for_uid( uid ) for uid in self.metagraph.uids.tolist() }
            for uid, meta in self.metadata.items():
                if self.stop_event.is_set(): return
                if meta == None or time.time() - meta['last_update'] >= UPDATE_TIMEOUT:
                    pretrain.utils.update_model_for_uid( uid, self.metagraph )
                    self.shouldeval[ uid ] = True

    def compute_losses_per_page( self, uid, batches_per_page: Dict[int, List[torch.Tensor]], pbar=None) -> Dict[int, List[float]]:

        try:
            # Load the pre-trained model from the specified path
            model_path = self.metadata[uid]['model_path']
            model = pretrain.model.get_model()
            model_weights = torch.load(model_path, map_location=torch.device(self.config.device))
            model.load_state_dict(model_weights)
            model.eval()  # Set the model to evaluation mode
            model.to(self.config.device)  # Move the model to the appropriate device
        except Exception as e:
            bt.logging.debug(f"Error loading model under path {model_path} with error: {e}")
            inf_losses = {}
            for page, batches in batches_per_page.items():
                inf_losses[page] = [math.inf for _ in batches]
            return inf_losses

        # Initialize a dictionary to store loss values for each page
        losses_per_page = {}

        # Iterate over each page and its corresponding batches
        for page, batches in batches_per_page.items():
            page_losses = []  # List to store losses for the current page

            # Process each batch and compute its loss
            for batch in batches:
                try:
                    # Perform a forward pass with the model to compute the loss
                    inputs = batch.to(self.config.device)
                    outputs = model(inputs, labels=inputs)
                    loss = outputs.loss.item()  # Get the scalar loss value
                    page_losses.append(loss)
                    if pbar is not None:
                        pbar.set_description(f"Loss: {uid} - {loss:.4f}")
                except Exception as e:
                    # Log the exception and append infinity to indicate failure
                    bt.logging.error(f"Exception occurred: {e}")
                    traceback.print_exc()  # Correctly print the stack trace
                    page_losses.append(math.inf)
            
            # Update the dictionary with the losses for the current page
            losses_per_page[page] = page_losses

        return losses_per_page
        
    def run_step( self ):
        """
            Executes a single validation step.
            - 1. Updates local models using wandb data.
            - 2. Generates random pages from Falcon Refined web for evaluation.
            - 3. Computes losses for each batch and each UID to attain losses per batch per page
            - 4. Determines the winning UID based on lowest average loss per batch.
            - 5. Logs win percentages for each UID to wandb and screen.
            - 6. Logs step results and updates weights and biases (wandb) if configured.

            Parameters:
            - wins_per_epoch (dict): A dictionary to record the number of wins per UID.
            - losses_per_epoch (dict): A dictionary to record all losses in epoch.
            - metagraph (object): An object representing the meta information of models.
            - wandb_step (int): The current step number for logging in weights and biases.

        """
        # Load metadata.
        self.metadata = { uid: pretrain.utils.load_metadata_for_uid( uid ) for uid in self.metagraph.uids.tolist() }

        # Get list of valid uids for step, valid uids must not be blacklisted 
        # and have a valid meta.
        uids = []
        for uid, meta in self.metadata.items():
            if meta != None and self.shouldeval[uid]: 
                uids.append( uid ) 
        bt.logging.success( f'Runnning step with uids: {uids}')

        # Generate random pages for evaluation and prepare batches for each page
        pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range(self.config.pages_per_eval)]
        batches_per_page = {
            page: list(pretrain.dataset.SubsetFalconLoader(
                batch_size = pretrain.batch_size,
                sequence_length = pretrain.sequence_length,
                pages = [page]
            )) for page in pages
        }
        total_batches = sum([ len(b) for b in batches_per_page.values()] )
        # Compute losses per page
        bt.logging.debug(f"computing losses on {uids}")
        losses_per_page_per_uid = { uid: None for uid in uids }
        pbar = tqdm( uids, desc="Loss", leave=False)
        for uid_i in pbar:
            losses = self.compute_losses_per_page( uid_i, batches_per_page, pbar )
            losses_per_page_per_uid[ uid_i ] = losses

        # Compute average loss per page
        average_loss_per_uid_per_page = { uid: {} for uid in uids }
        for uid_i in uids:
            for page_j in pages:
                losses = losses_per_page_per_uid[ uid_i ][ page_j ]
                average_loss = sum(losses)/len(losses)
                average_loss_per_uid_per_page[ uid_i ][ page_j ] = average_loss
                if uid_i in self.losses_per_epoch:
                    self.losses_per_epoch[ uid_i ].append(average_loss)
                else:
                    self.losses_per_epoch[ uid_i ] = [average_loss]

        # Compute best average loss
        best_average_loss = math.inf
        best_average_loss_uid = None
        for uid_i in uids:
            average_loss = sum([ average_loss_per_uid_per_page[uid_i][page_j] for page_j in pages ] ) / len( pages )
            if average_loss < best_average_loss:
                best_average_loss = average_loss
                best_average_loss_uid = uid_i


        # Compute wins.
        def better( i, j, p, b ):
            il = losses_per_page_per_uid[ i ][ p ][ b ]
            jl = losses_per_page_per_uid[ j ][ p ][ b ]
            it = self.metadata[ i ]['timestamp']
            jt = self.metadata[ i ]['timestamp']
            il = (1 - 0.03) * il if it < jt else il
            jl = (1 - 0.03) * jl if it < jt else jl
            if il < jl: return True
            else: return False

        wins = { uid: 0 for uid in uids }
        win_rate = { uid: 0 for uid in uids }
        for i in uids:
            total_matches = 0
            for j in uids:
                for p in pages:
                    for b, _ in enumerate( batches_per_page[page_j] ):
                        wins[i] += 1 if better( i, j, p, b ) else 0
                        total_matches += 1
            win_rate[ i ] = wins[ i ] / total_matches
            self.wins_per_epoch[i] = self.wins_per_epoch.get(i, 0) + wins[i]
        total_matches /= 2

        # Build step log
        step_log = {
            'timestamp': time.time(),
            'pages': pages,
            'uids': uids,
            'best_average_loss': best_average_loss,
            'best_average_loss_uid': best_average_loss_uid,
            'uid_data': {}
        }
        for uid in uids:
            try:
                average_losses = [average_loss_per_uid_per_page[uid][pagek] for pagek in pages]
                average_loss = sum(average_losses) / len(average_losses)
                step_log['uid_data'][ str(uid) ] = {
                    'uid': uid,
                    'runid': self.metadata[ uid ]['runid'],
                    'timestamp': self.metadata[ uid ]['timestamp'],
                    'last_update': self.metadata[ uid ]['last_update'],
                    'average_losses': average_losses,
                    'average_loss': average_loss,
                    'win_rate': win_rate[ uid ],
                    'win_total': wins[ uid ],
                }

            except:
                continue
        table = Table(title="Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("average_loss", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("last_update", style="magenta")
        table.add_column("timestamp", style="magenta")
        table.add_column("wins_per_epoch", style="magenta")
        table.add_column("blacklist", style="magenta")
        for uid in uids:
            table.add_row(
                str(uid), 
                str(step_log['uid_data'][ str(uid) ]['average_loss']), 
                str(step_log['uid_data'][ str(uid) ]['win_rate']),
                str(step_log['uid_data'][ str(uid) ]['win_total']),
                str(step_log['uid_data'][ str(uid) ]['last_update']),
                str(step_log['uid_data'][ str(uid) ]['timestamp']),
                str( self.wins_per_epoch[uid] ),
                str( not self.shouldeval[ uid ]),
            )
        console = Console()
        console.print(table)

        # Sink step log.
        bt.logging.trace(f"Step results: {step_log}")
        original_format_json = json.dumps(step_log)
        uids = step_log['uids']
        uid_data = step_log['uid_data']

        # Create a new dictionary with the required format
        graphed_data = {
            'uid_data': {str(uid): uid_data[str(uid)]['average_loss'] for uid in uids}
        }
        if self.config.wandb.on: wandb.log({ **graphed_data, "original_format_json": original_format_json}, step=self.global_step)

    def set_weights( self ):
        """
            Completes the validation epoch determining the weights that need to be set on chain
            and firing off the extrinsic. 

            Parameters:
            - wins_per_epoch (dict): A dictionary to record the number of wins per UID.
            - wandb_step (int): The current step number for logging in weights and biases.
        """
        # === Compute weights from wins ===
        weights = torch.zeros( len(self.metagraph.hotkeys) )
        add = 0.05
        total_add = len( list( self.wins_per_epoch.keys() ) ) * add
        total_wins = sum(self.wins_per_epoch.values())  # Sum of all wins
        for uid in self.wins_per_epoch:
            weights[uid] = (self.wins_per_epoch[uid] + add) / (total_wins + total_add)

        # === Set weights ===
        self.subtensor.set_weights(
            netuid = pretrain.NETUID,
            wallet = self.wallet,
            uids = self.metagraph.uids,
            weights = weights,
            wait_for_inclusion=False,
        )
  
        sorted_weights = sorted(enumerate(weights, start=0), key=lambda x: x[1], reverse=True)
        # Filter to keep only positive weights
        positive_weights = [(uid, weight) for uid, weight in sorted_weights if weight > 0]
        # Limit to top 20 (or less if there are not enough positive weights)
        positive_weights = positive_weights[:20]
        table = Table(title="Top Weights")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for uid, value in positive_weights:
            table.add_row(str(uid), str(value.item()))
        console = Console()
        console.print(table)

    def run(self):
        while True:
            try:
                # Init a new dict for counters.
                self.wins_per_epoch = {}
                self.losses_per_epoch = {}
            
                while self.metagraph.block.item() - self.last_epoch < self.config.blocks_per_epoch:
                    self.run_step()
                    self.metagraph = self.subtensor.metagraph( pretrain.NETUID )
                    bt.logging.debug(f"{self.metagraph.block.item() - self.last_epoch } / {self.config.blocks_per_epoch} blocks until next epoch.")
                    self.global_step += 1

                # Finish epoch.
                self.set_weights( )
                self.last_epoch = self.metagraph.block.item()
                self.epoch_step += 1

            except KeyboardInterrupt:
                bt.logging.info("KeyboardInterrupt caught, gracefully closing the wandb run...")
                if self.config.wandb.on: self.wandb_run.finish()
                exit()

            except Exception as e:
                bt.logging.error(f"Error in validator loop \n {e} \n {traceback.format_exc()}")


if __name__ == "__main__":
    Validator().run()