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
import bittensor as bt
from tqdm import tqdm
from typing import Dict, List
from rich.table import Table
from rich.console import Console

# Global artifact name
UPDATE_TIMEOUT = 60*60*6
ARTIFACT_NAME:str = "model.pth"

class Validator:

    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument( "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")
        parser.add_argument( '--wandb.off', dest = 'wandb.on', action='store_false', help='Turn off wandb logging.' )
        parser.add_argument( '--blocks_per_epoch', type=int, default=360, help='Number of blocks to wait before setting weights.' )
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
            wandb_run =  wandb.init(
                name = run_name,
                anonymous = "allow",
                reinit = False,
                project = 'openpretraining',
                entity = 'opentensor-dev',
                config = self.config,
                dir = self.config.full_path,
            )
            # Sign wandb run.
            wandb.init( project = "openpretraining", entity="opentensor-dev" )
            self.config.signature = self.wallet.hotkey.sign( wandb_run.id.encode() ).hex()
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

        # === Running args ===
        self.epoch_step = 0 
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()
        self.last_update_check = {}
        self.shouldeval = { uid: uid in self.metagraph.I.topk(10).indices.tolist() for uid in self.metagraph.uids.tolist()  }
        self.metadata = { uid: pretrain.utils.load_metadata_for_uid( uid ) for uid in self.metagraph.uids.tolist() }

    def update_models( self ):
        return
        # Go through sorted metadata, if the update interval has passed, update the model.
        self.metadata = { uid: pretrain.utils.load_metadata_for_uid( uid ) for uid in self.metagraph.uids.tolist() }
        pbar = tqdm( self.metadata.items(), desc="Updating", leave=False)

        # Iterate through models checking to see if we should attempt and update.
        for uid, meta in pbar:
            pbar.set_description(f"Updating uid: {uid}")
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
        pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range(pretrain.n_eval_pages)]
        batches_per_page = {
            page: list(pretrain.dataset.SubsetFalconLoader(
                batch_size = pretrain.batch_size,
                sequence_length = pretrain.sequence_length,
                pages = [page]
            )) for page in pages
        }
        total_batches = sum([ len(b) for b in batches_per_page.values()] )
        bt.logging.trace(f"pages: {pages}")

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

        # Function returns True if this uid has lowest loss across all other uids on this 
        # batch, in case of ties takes uid with better timestamp.
        def is_win( this_uid, other_uid, page_j, batch_k ):
            this_loss = losses_per_page_per_uid[ this_uid ][ page_j ][ batch_k ]
            other_loss = losses_per_page_per_uid[ other_uid ][ page_j ][ batch_k ]
            print( this_uid, other_uid, this_loss, other_loss)
            if this_loss >= other_loss:
                return False
            else:
                return True
            # this_timestamp = self.metadata[ this_uid ]['timestamp']
            # other_timestamp = self.metadata[ other_uid ]['timestamp']
            # if this_timestamp > other_timestamp:
            #     other_loss *= (1 - 0.03)
            # elif this_timestamp < other_timestamp:
            #     this_loss *= (1 - 0.03)
            # if this_loss > other_loss:
            #     return True
            # else:
            #     return False
        def is_winning_loss_with_timestamps( this_uid, page_j, batch_k ):
            for other_uid in uids:
                if not is_win( this_uid, other_uid, page_j, batch_k ):
                    bt.logging.success(f'Failed {this_uid} {other_uid} {page_j} {batch_k} {losses_per_page_per_uid[ this_uid ][ page_j ][ batch_k ]} {losses_per_page_per_uid[ other_uid ][ page_j ][ batch_k ]}')
                    return False
            return True

        # Compute total wins per uid per page 
        total_wins_per_uid_per_page = { uid: { page: 0 for page in pages } for uid in uids }
        for this_uid in uids:
            self.wins_per_epoch[ this_uid ] = 0
            for page_j in pages:
                for batch_k, _ in enumerate( batches_per_page[page_j] ):
                    if is_winning_loss_with_timestamps( this_uid, page_j, batch_k ):
                        total_wins_per_uid_per_page[ this_uid ][ page_j ] += 1
                        self.wins_per_epoch[ this_uid ] += 1 

            # We will not recheck if a miner has zero wins until the model is reupdated.
            if self.wins_per_epoch[ this_uid ] == 0:
                self.shouldeval[ this_uid ] = False

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
                win_rate = sum ( [total_wins_per_uid_per_page[ uid ][ pagek ] for pagek in pages ]) / total_batches
                win_total = sum ( [total_wins_per_uid_per_page[ uid ][ pagek ] for pagek in pages ])
                step_log['uid_data'][ str(uid) ] = {
                    'uid': uid,
                    'runid': self.metadata[ uid ]['runid'],
                    'timestamp': self.metadata[ uid ]['timestamp'],
                    'average_losses': average_losses,
                    'average_loss': average_loss,
                    'win_rate': win_rate,
                    'win_total': win_total,
                }
            except:
                continue
        print ('total_batches',total_batches)
        print ('total_wins_per_uid_per_page',total_wins_per_uid_per_page)
        print ('average_loss_per_uid_per_page',average_loss_per_uid_per_page)
        print ('losses_per_page_per_uid',average_loss_per_uid_per_page)
        table = Table(title="Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("average_losses", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        for uid in uids:
            table.add_row(
                str(uid), 
                str(step_log['uid_data'][ str(uid) ]['average_loss']), 
                str(step_log['uid_data'][ str(uid) ]['win_rate']),
                str(step_log['uid_data'][ str(uid) ]['win_total'])
            )
        console = Console()
        console.print(table)

        # Sink step log.
        bt.logging.success(f"Step results: {step_log}")
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
        for uid in self.wins_per_epoch:
            total_wins = sum(self.wins_per_epoch.values())  # Sum of all wins
            weights[uid] = (self.wins_per_epoch[uid] + add) / (total_wins + total_add)

        # === Set weights ===
        self.subtensor.set_weights(
            netuid = pretrain.NETUID,
            wallet = self.wallet,
            uids = self.metagraph.uids,
            weights = weights,
            wait_for_inclusion=False,
        )
  
        sorted_weights = sorted(enumerate(weights, start=1), key=lambda x: x[1], reverse=True)[:10]
        table = Table(title="Top10 Weights")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for uid, value in sorted_weights:
            table.add_row(str(uid), str(value.item()))
        console = Console()
        console.print(table)

    def run(self):
        while True:
            try:
                # Init a new dict for counters.
                self.wins_per_epoch = {}
                self.losses_per_epoch = {}
                self.update_models()
            
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