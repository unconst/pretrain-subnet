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
import asyncio
import argparse
import pretrain
import traceback
import threading
import multiprocessing
import bittensor as bt
from tqdm import tqdm
from typing import Dict, List
from rich.table import Table
from rich.console import Console

# Global artifact name
UPDATE_TIMEOUT = 60*60*2
ARTIFACT_NAME:str = "model.pth"
RUN_STEP_MAX_TIME = 60 * 20 # 20 min run step timeout.
class Validator:

    @staticmethod
    def config():
        parser = argparse.ArgumentParser()
        parser.add_argument( "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")
        parser.add_argument( '--wandb.off', dest = 'wandb.on', action='store_false', help='Turn off wandb logging.' )
        parser.add_argument( '--blocks_per_epoch', type=int, default=360, help='Number of blocks to wait before setting weights.' )
        parser.add_argument( '--pages_per_eval', type=int, default=3, help='Number of pages used to eval each step.' )
        parser.add_argument( '--sample_min', type=int, default=30, help='Number of uids to eval each step.' )
        parser.add_argument( '--reset_wandb', action='store_true', help='Creates a new wandb run instead of using an older on.' )
        parser.add_argument( '--dont_set_weights', action='store_true', help='Creates a new wandb run instead of using an older on.' )
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
            import json
            run_id_file = self.config.full_path + '/run.json'
            try:
                if self.config.reset_wandb: raise Exception('go to create new run.')
                with open( run_id_file, 'r' ) as f:
                    self.run_id = json.load( f )['WANDB_RUN_ID']
                    bt.logging.success(f'Continuing run, loaded run_id: {self.run_id}')
            except Exception as e: 
                self.run_id = wandb.util.generate_id()
                bt.logging.success(f'First run, creating new run_id: {self.run_id} {e}')
            with open( run_id_file, 'w' ) as f:
                json.dump({'WANDB_RUN_ID': self.run_id}, f)
                bt.logging.success(f'Saved: {self.run_id} to file.')
            self.run_name = f'validator-{self.uid}-' + ''.join(random.choice( string.ascii_uppercase + string.digits ) for i in range(10))
            self.config.uid = self.uid
            self.config.hotkey = self.wallet.hotkey.ss58_address
            self.config.run_name = self.run_name
            self.config.type = "validator"
            self.config.version = pretrain.__version__
            self.wandb_run = wandb.init(
                id = self.run_id,
                name = self.run_name,
                anonymous = "allow",
                reinit = False,
                project = pretrain.WANDB_PROJECT,
                entity = 'opentensor-dev',
                config = self.config,
                dir = self.config.full_path,
            )
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
        if self.wallet.hotkey.ss58_address not in self.metagraph.hotkeys: raise Exception(f"You are not registered. Use `btcli s register --netuid {pretrain.NETUID}` to register.")
        self.uid = self.metagraph.hotkeys.index( self.wallet.hotkey.ss58_address )
        bt.logging.success( f'You are registered with address: {self.wallet.hotkey.ss58_address} and uid: {self.uid}' )
        self.init_wandb()

        # === Running args ===
        self.weights = torch.zeros_like(self.metagraph.S)
        self.epoch_step = 0 
        self.global_step = 0
        self.last_epoch = self.metagraph.block.item()
        self.last_update_check = {}

        # Metadata info.
        self.model_metadata = { uid: pretrain.utils.load_metadata_for_uid( uid ) for uid in self.metagraph.uids.tolist() }
        self.delta_metadata = { uid: pretrain.utils.load_delta_metadata_for_uid( uid ) for uid in self.metagraph.uids.tolist() }

        # Uids to eval per step sets.
        self.model_uids_to_eval = set()
        self.delta_uids_to_eval = set()
        for uid in self.metagraph.uids.tolist():
            if self.model_metadata[ uid ] != None:
                self.model_uids_to_eval.add( uid )
            if self.delta_metadata[ uid ] != None:
                self.delta_uids_to_eval.add( uid )

        # == Initialize the update thread ==
        self.stop_event = threading.Event()
        self.update_thread = threading.Thread(target=self.update_models, daemon=True)
        self.update_thread.start()

    def __del__(self):
        if hasattr( self, 'stop_event'):
            self.stop_event.set()
            self.update_thread.join()

    def update_models( self ):
        # The below loop iterates across all miner uids and checks to see 
        # if they should be updated.
        last_uid_update = -1
        while not self.stop_event.is_set():
            if self.stop_event.is_set(): return
            block = self.subtensor.block 
            uid = block % 256
            if uid == last_uid_update: 
                time.sleep(1)
                continue
            last_uid_update = uid
            bt.logging.success( f'Updating model under uid: {uid} for block: {block}')
            pretrain.utils.update_model_for_uid( uid, self.metagraph )
            pretrain.utils.update_delta_for_uid( uid, self.metagraph )
            self.model_uids_to_eval.add( uid )
            self.delta_uids_to_eval.add( uid )
            bt.logging.trace(f'uids to eval add: {uid}')
 
    async def try_set_weights( self, ttl: int ):
        async def _try_set_weights():
            try:
                self.weights.nan_to_num( 0.0 )
                self.subtensor.set_weights(
                    netuid = pretrain.NETUID,
                    wallet = self.wallet,
                    uids = self.metagraph.uids,
                    weights = self.weights,
                    wait_for_inclusion=False,
                )
            except: pass
            ws, ui = self.weights.topk( len( self.weights ) )
            table = Table(title="All Weights")
            table.add_column("uid", justify="right", style="cyan", no_wrap=True)
            table.add_column("weight", style="magenta")
            for index, weight in list(zip(ui.tolist(), ws.tolist())):
                table.add_row(str(index), str(round(weight, 4)))
            console = Console()
            console.print(table)
        try:
            bt.logging.debug(f'Setting weights.') 
            await asyncio.wait_for( _try_set_weights() , ttl )
            bt.logging.debug(f'Finished setting weights.') 
        except asyncio.TimeoutError: 
            bt.logging.error(f'Failed to set weights after {ttl} seconds')

    async def try_sync_metagraph( self, ttl: int ):
        def sync_metagraph( endpoint ):
            metagraph = bt.subtensor( endpoint ).metagraph( pretrain.NETUID )
            metagraph.save()
        process = multiprocessing.Process(target=sync_metagraph, args=( self.subtensor.chain_endpoint, ))
        process.start()
        process.join(timeout=ttl)
        if process.is_alive():
            process.terminate()
            process.join()
            bt.logging.error(f'Failed to sync metagraph after {ttl} seconds')
        self.metagraph.load()

    async def try_run_step( self, ttl: int ):
        async def _try_run_step():
            await self.run_step()
        try: 
            bt.logging.debug(f'Running step.') 
            await asyncio.wait_for( _try_run_step() , ttl )
            bt.logging.debug(f'Finished running step.') 
        except asyncio.TimeoutError: 
            bt.logging.error(f'Failed to run step after {ttl} seconds')

    # Add a 20 minute max timeout.
    async def run_step( self ):

        # Load metadata.
        self.model_metadata = { uid: pretrain.utils.load_metadata_for_uid( uid ) for uid in self.metagraph.uids.tolist() }
        self.delta_metadata = { uid: pretrain.utils.load_delta_metadata_for_uid( uid ) for uid in self.metagraph.uids.tolist() }

        # Filter model uids to eval
        model_uids = []
        for uid in list( self.model_uids_to_eval ):
            if self.metadata[uid] == None: continue
            if pretrain.utils.check_run_exists( uid, self.model_metadata[uid], self.metagraph ): model_uids.append( uid )
            else: bt.logging.debug( f'uid:{uid} run does not exist or is not valid, removing from uids to eval.')

        # Filter delta uids to eval
        delta_uids = [] 
        for uid in list( self.delta_uids_to_eval ):
            if self.metadata[uid] == None: continue
            if pretrain.utils.check_run_exists( uid, self.delta_metadata[uid], self.metagraph ): delta_uids.append( uid )
            else: bt.logging.debug( f'uid:{uid} run does not exist or is not valid, removing from uids to eval.')
        random.shuffle( uids )
        bt.logging.success( f'Runnning step with model uids: {model_uids}, delta uids: {delta_uids}')

        # Generate random pages for evaluation and prepare batches for each page
        # the dataset contains >900 million pages to eval over.
        pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range(self.config.pages_per_eval)]
        batches = {
            page: list(pretrain.dataset.SubsetFalconLoader(
                batch_size = pretrain.batch_size,
                sequence_length = pretrain.sequence_length,
                pages = [page]
            )) for page in pages
        }

        # Compute model scoring.
        model_losses_per_uid = { muid: None for muid in model_uids } 
        for muid_i in model_uids:
            model = pretrain.model.get_model_for_uid( muid_i )
            losses = pretrain.validation.compute_losses( model, batches, device = self.config.device )
            model_losses_per_uid[ muid_i ] = losses

        # Compute delta scoring.
        base_uid = max(range(256), key=lambda uid: self.metagraph.I[uid].item())
        base_model = pretrain.model.get_model_for_uid( base_uid, device = self.config.device  )
        delta_losses_per_uid = { duid: None for duid in delta_uids } 
        for duid_i in delta_uids:
            delta = pretrain.model.get_delta_for_uid( duid_i )
            shifted_base_model = pretrain.model.apply_delta( base_model, delta )
            losses = pretrain.validation.compute_losses( shifted_base_model, batches, device = self.config.device )
            delta_losses_per_uid[ duid_i ] = losses

        # Compute model win rates + weights
        model_wins, model_win_rate = pretrain.validation.compute_wins( model_losses_per_uid, batches, self.model_metadata )
        model_weights = torch.tensor([ model_win_rate[ uid ] for uid in model_uids ], dtype=torch.float32)
        self.model_weights = torch.softmax( model_weights / pretrain.temperature, dim=0 )

        # Compute delta win rates + weights.
        delta_wins, delta_win_rate = pretrain.validation.compute_wins( delta_losses_per_uid, batches, self.delta_metadata )        
        delta_weights = torch.tensor([ delta_win_rate[ uid ] for uid in delta_uids ], dtype=torch.float32)
        self.delta_weights = torch.softmax( delta_weights / pretrain.temperature, dim=0 )

        # Sum weights.
        new_weights = self.weights.clone()
        for duid in delta_uids: new_weights[ duid ] += self.delta_weights[ duid ]
        for muid in model_uids: new_weights[ muid ] += self.model_weights[ muid ]
        new_weights /= new_weights.sum()

        # Moving average the weights.
        self.weights = pretrain.alpha * self.weights + ( 1 - pretrain.alpha ) * new_weights
        self.weights.nan_to_num( 0.0 )

        # Blacklist bad miners. Here we remove uids from eval set 
        # based on their win rate, this prunes miners down the sample min
        # miner uids are replaced when their model is updated on wandb after a timeout.
        for uid in model_uids:
            if len( list(self.model_uids_to_eval) ) <= self.config.sample_min: break
            if model_win_rate[uid] < 0.5: 
                self.model_uids_to_eval.remove( uid )

        for uid in delta_uids:
            if len( list(self.delta_uids_to_eval) ) <= self.config.sample_min: break
            if delta_win_rate[uid] < 0.5: 
                self.delta_uids_to_eval.remove( uid )
        
        # Build step log
        step_log = {
            'timestamp': time.time(),
            'pages': pages,
            'uids': uids,
            'model_uid_data': {},
            'delta_uid_data': {}
        }
        for uid in model_uids:
            try:
                model_page_average_losses = [ sum( model_losses_per_uid[ uid ][ pagek ])/ len( model_losses_per_uid[ uid ][ pagek ] ) for pagek in pages]
                model_average_loss = sum(model_page_average_losses) / len(model_page_average_losses)
                step_log['model_uid_data'][ str(uid) ] = {
                    'model_uid': uid,
                    'model_runid': self.model_metadata[ uid ]['runid'],
                    'model_timestamp': self.model_metadata[ uid ]['timestamp'],
                    'model_last_update': self.model_metadata[ uid ]['last_update'],
                    'model_average_losses': model_page_average_losses,
                    'model_average_loss': model_average_loss,
                    'model_win_rate': model_win_rate[ uid ],
                    'model_wins': model_wins[ uid ],
                    'model_weight': self.model_weights[ uid ].item()
                }
            except:
                continue
        for uid in delta_uids:
            try:
                delta_page_average_losses = [ sum( delta_losses_per_uid[ uid ][ pagek ])/ len( delta_losses_per_uid[ uid ][ pagek ] ) for pagek in pages]
                delta_average_loss = sum(delta_page_average_losses) / len(delta_page_average_losses)
                step_log['delta_uid_data'][ str(uid) ] = {
                    'delta_uid': uid,
                    'delta_runid': self.delta_metadata[ uid ]['runid'],
                    'delta_timestamp': self.delta_metadata[ uid ]['timestamp'],
                    'delta_last_update': self.delta_metadata[ uid ]['last_update'],
                    'delta_average_losses': delta_page_average_losses,
                    'delta_average_loss': delta_average_loss,
                    'delta_win_rate': delta_win_rate[ uid ],
                    'delta_wins': delta_wins[ uid ],
                    'delta_weight': self.delta_weights[ uid ].item()
                }
            except:
                continue
        table = Table(title="Model Step")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("average_loss", style="magenta")
        table.add_column("win_rate", style="magenta")
        table.add_column("win_total", style="magenta")
        table.add_column("weights", style="magenta")
        table.add_column("last_update", style="magenta")
        table.add_column("timestamp", style="magenta")
        for uid in uids:
            table.add_row(
                str(uid), 
                str( round(step_log['uid_data'][ str(uid) ]['average_loss'], 4)), 
                str( round(step_log['uid_data'][ str(uid) ]['win_rate'], 4)),
                str(step_log['uid_data'][ str(uid) ]['win_total']),
                str( round(self.weights[uid].item(), 4) ),
                str( round(step_log['uid_data'][ str(uid) ]['last_update'], 0)),
                str( step_log['uid_data'][ str(uid) ]['timestamp']),
            )
        console = Console()
        console.print(table)

        ws, ui = self.weights.topk( len( self.weights ))
        table = Table(title="Weights > 0.001")
        table.add_column("uid", justify="right", style="cyan", no_wrap=True)
        table.add_column("weight", style="magenta")
        for index, weight in list(zip(ui.tolist(), ws.tolist())):
            if weight > 0.001:
                table.add_row(str(index), str(round(weight, 4)))
        console = Console()
        console.print(table)

        # Sink step log.
        bt.logging.trace(f"Step results: {step_log}")
        original_format_json = json.dumps(step_log)
        uids = step_log['uids']
        uid_data = step_log['uid_data']

        # Create a new dictionary with the required format
        graphed_data = {
            'time': time.time(),
            'block': self.subtensor.block,
            'uid_data': {str(uid): uid_data[str(uid)]['average_loss'] for uid in uids},
            'weight_data': {str(uid): self.weights[uid].item() for uid in uids}
        }

        if self.config.wandb.on: 
            bt.logging.trace('Logging to Wandb')
            self.wandb_run.log({ **graphed_data, "original_format_json": original_format_json}, step=self.global_step)
            bt.logging.trace('finished log to Wandb')
        bt.logging.debug('Finished run step.')

    async def run(self):
        while True:
            try:            
                while self.metagraph.block.item() - self.last_epoch < self.config.blocks_per_epoch:
                    await self.try_run_step( ttl = RUN_STEP_MAX_TIME )
                    await self.try_sync_metagraph( ttl = 60 )
                    bt.logging.debug(f"{self.metagraph.block.item() - self.last_epoch } / {self.config.blocks_per_epoch} blocks until next epoch.")
                    self.global_step += 1

                if not self.config.dont_set_weights:
                    await self.try_set_weights( ttl = 60 )
                self.last_epoch = self.metagraph.block.item()
                self.epoch_step += 1

            except KeyboardInterrupt:
                bt.logging.info("KeyboardInterrupt caught, gracefully closing the wandb run...")
                if self.config.wandb.on: self.wandb_run.finish()
                exit()

            except Exception as e:
                bt.logging.error(f"Error in validator loop \n {e} \n {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run( Validator().run() ) 