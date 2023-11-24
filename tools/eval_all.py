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

import torch
import random
import argparse
from tqdm import tqdm
import bittensor as bt
import pretrain as pt
from rich.table import Table
from rich.console import Console

parser = argparse.ArgumentParser()
parser.add_argument( "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")        
parser.add_argument( '--pages_per_eval', type=int, default=3, help='Number of pages used to eval each step.' )
parser.add_argument( '--no_sync', action='store_true', help='Does not resync runs.' )
bt.logging.add_args(parser)
config = bt.config(parser)
bt.logging( config = config )

# Sync graph
metagraph = bt.metagraph( pt.NETUID )

uids = []
timestamps = {}
for uid in metagraph.uids.tolist():
    if not config.no_sync:
        pt.graph.sync( uid, metagraph )
    if pt.graph.timestamp( uid ) != None:
        uids.append( uid )
        timestamps[uid] = pt.graph.timestamp( uid )

# Get all batches.
pages = [random.randint(1, pt.dataset.SubsetFalconLoader.max_pages) for _ in range(config.pages_per_eval)]
batches = list( pt.dataset.SubsetFalconLoader( batch_size = pt.batch_size, sequence_length = pt.sequence_length, pages = pages) )

# Compute losses
losses = {}
for uid in tqdm( uids, desc = 'Computing losses' ):
    # get model or uid or None if we have not synced the model.
    model = pt.graph.model( uid, device = config.device )
    if model != None: 
        losses[uid] = pt.validation.compute_losses( model, batches, device = config.device )

# Compute wins
wins = { uid: 0 for uid in losses.keys() }
win_rate = { uid: 0 for uid in losses.keys() }
for i, uid_i in enumerate( losses.keys() ):
    total_matches = 0
    for j, uid_j in enumerate( losses.keys() ):
        if i == j: continue
        for batch_idx, _ in enumerate( batches ):
            loss_i = losses[ uid_i ][ batch_idx ]
            loss_j = losses[ uid_j ][ batch_idx ] 
            time_i = timestamps[ uid_i ]
            time_j = timestamps[ uid_j ] 
            wins[ uid_i ] += 1 if pt.validation.iswin( loss_i, loss_j, time_i, time_j ) else 0
            total_matches += 1
    win_rate[ uid_i ] = wins[ uid_i ] / total_matches if total_matches > 0 else 0


table = Table(title="Step")
table.add_column("uid", justify="right", style="cyan", no_wrap=True)
table.add_column("average_loss", style="magenta")
table.add_column("win_rate", style="magenta")
table.add_column("win_total", style="magenta")
table.add_column("timestamp", style="magenta")
for uid in losses.keys():
    table.add_row(
        str( uid ), 
        str( round( sum(losses[ uid ])/len(batches), 4)), 
        str( round( win_rate[ uid ], 4)),
        str( wins[ uid ]),
        str( timestamps[ uid ]),
    )
console = Console()
console.print(table)

