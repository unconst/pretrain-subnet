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

import math
import time
import torch
import random
import pretrain
import bittensor as bt
from rich import print
from rich.text import Text
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns

async def run_eval_on_state( self, eval_state ) -> float:

    # Load the eval state into the model.
    eval_model = pretrain.model.get_model().to( self.config.device )
    eval_model.load_state_dict( eval_state )
    eval_model.zero_grad()
    eval_model.train()

    # Set up dataloader
    random_pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages)]
    loader = pretrain.dataset.SubsetFalconLoader(
        batch_size = 3,
        sequence_length = 512,
        pages = random_pages
    )
    torch.backends.cudnn.benchmark = True

    # Run eval
    n_batches = 0
    average_loss = 0.0
    for i, batch in enumerate( loader ):
        inputs = batch.to( eval_model.device )
        outputs = eval_model( inputs, labels=inputs )
        outputs.loss.backward()
        average_loss += outputs.loss.detach().item()
        n_batches += 1
        torch.cuda.empty_cache()
        bt.logging.success( f'Acc: step: {i} loss: {outputs.loss}' )

    # Return average loss on batch.
    return average_loss / n_batches

# === Training loop ===
async def foreground_loop(self: object):    

    while True:

        forward_event = {}

        # Get uid from serving online miner.
        # available_uids = [ uid.item() for uid in self.metagraph.uids if self.metagraph.axons[uid].is_serving and (self.metagraph.block.item() - self.metagraph.last_update[uid] < 1000) ]
        available_uids = [230]
        if len( available_uids ) == 0:
            raise Exception('No available uids.')
        uid = random.choice( available_uids )
        forward_event['uid'] = uid

        # Build dendrite and query random miner.
        start_query = time.time()
        dendrite = bt.dendrite(wallet=self.wallet)
        synapse = pretrain.protocol.GetState()
        response = await dendrite.forward( self.metagraph.axons[uid], synapse, timeout=360, deserialize=False )
        await dendrite.close_session()        
        forward_event['query_time'] = time.time() - start_query

        # Response was a failure.
        if not response.is_success:
            forward_event['success'] = False
            forward_event['better'] = False
            forward_event['score'] = -1

        else:
            # Get model state and run random eval.
            eval_state = response.deserialize()
            eval_loss = await run_eval_on_state( self, eval_state )
            forward_event['eval_loss'] = eval_loss

            if eval_loss < self.best_average_loss:
                self.best_average_loss = eval_loss
                self.best_model_state = eval_state
                forward_event['success'] = True
                forward_event['better'] = True
                forward_event['score'] = 1

            else:
                forward_event['success'] = True
                forward_event['better'] = False
                forward_event['score'] = 0

        update_global_state( self, forward_event )

def update_global_state( self, forward_event: dict ):

    # Update score for uid.
    uid = forward_event['uid']
    if uid not in self.global_state['uid_state']:
        self.global_state['uid_state'][uid] = {
            'score': -1,
            'n_successes': 0,
            'n_forward': 0,
        }
    self.global_state['uid_state'][uid]['score'] = self.config.alpha * self.global_state['uid_state'][uid]['score'] + ( 1 - self.config.alpha ) * forward_event['score'] 
    self.global_state['uid_state'][uid]['n_forward'] += + 1
    if forward_event['success']:
        self.global_state['uid_state'][uid]['n_successes'] += + 1 

    # Log UID state
    items = [
        Text(f"UID: {uid}, Score: {math.exp(self.global_state['uid_state'][uid]['score'])}:{self.global_state['uid_state'][uid]['n_successes']}/{self.global_state['uid_state'][uid]['n_forward']}" )
        for uid in self.global_state['uid_state'].keys()
    ]
    columns = Columns(items, equal=True, expand=True)
    panel = Panel(columns, title="Scores")
    print( panel )

    # Log the forward event to the console
    self.global_state['n_steps'] += 1
    self.global_state['n_successes'] += 1 if 'success' in forward_event and forward_event['success'] else 0
    self.global_state['n_failures'] += 1 if 'success' in forward_event and not forward_event['success'] else 0
    self.global_state['steps_per_second'] = 1 / (time.time() - self.global_state['last_query'])
    self.global_state['last_query'] = time.time()
    self.global_state['eval_loss'] = forward_event['eval_loss'] if 'eval_loss' in forward_event else self.global_state['eval_loss']
    self.global_state['query_time'] = forward_event['query_time'] if 'query_time' in forward_event else self.global_state['query_time']

    # Create a log dictionary
    log = {
        'uid': forward_event['uid'],
        'n_steps': self.global_state['n_steps'],
        'n_successes': self.global_state['n_successes'],
        'n_failures': self.global_state['n_failures'],
        'steps_per_second': self.global_state['steps_per_second'],
        'eval_loss': self.global_state['eval_loss'],
        'query_time': self.global_state['query_time'],
    }

    # Log State
    table = Table()
    table.add_column("Metric", style="bold magenta")
    table.add_column("Value", style="bold green")
    table.add_row("Steps", str(log['n_steps']))
    table.add_row("Successes", str(log['n_successes']))
    table.add_row("Failures", str(log['n_failures']))
    table.add_row("Steps Per Second", f"{log['steps_per_second']:.2f}")
    table.add_row("Query Time", f"{log['query_time']:.2f}")
    table.add_row("Eval Loss", f"{log['eval_loss']:.2f}")
    print(table)


    # Log the forward event to wandb if configured
    if self.wandb:
        self.wandb.log( log )







