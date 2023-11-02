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

import wandb
import torch
import random
import pretrain
import argparse
import bittensor as bt


def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument( "--model_path", type = 'str', help="Run name.", required=True )
    bt.subtensor.add_args(parser)
    bt.logging.add_args(parser)
    bt.wallet.add_args(parser)
    bt.axon.add_args(parser)
    config = bt.config(parser)
    return config
config = get_config()

wallet = bt.wallet( config = config )
subtensor = bt.subtensor( config = config )
dendrite = bt.dendrite( wallet = wallet )
metagraph = subtensor.metagraph( pretrain.NETUID )
torch.backends.cudnn.benchmark = True

while True:

    api = wandb.Api( timeout = 100 )

    random_pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages)]
    loader = pretrain.dataset.SubsetFalconLoader(
        batch_size = 3,
        sequence_length = 512,
        pages = random_pages
    )

    best_average_loss = None
    best_uid = None
    best_uid_run = None
    for uid in metagraph.uids:

        axon = metagraph.axon[uid]
        run_name = dendrite.query( axon, pretrain.GetRun() ).run_name
        run = api.run("opentensor-dev/openpretraining/{run_name}")

        hotkey = run.config.get('hotkey')
        if hotkey != metagraph.hotkey[uid]:
            raise ValueError("Hotkey mismatch")
        
        # Download the model weights
        artifact_name = "model_weights.pth"
        run.file(artifact_name).download(replace=True)

        model = pretrain.model.get_model()
        model_weights = torch.load(artifact_name)
        model.load_state_dict(model_weights)
        model.zero_grad()
        model.train()
        model.to( 'cuda' )

        # Run eval
        average_loss = 0.0
        for i, batch in enumerate( loader ):
            inputs = batch.to( model.device )
            outputs = model( inputs, labels=inputs )
            outputs.loss.backward()
            average_loss += outputs.loss.detach().item()
            torch.cuda.empty_cache()
            bt.logging.success( f'Acc: step: {i} loss: {outputs.loss}' )

        # Get best average loss and best uid
        # ties broken on timestamp.
        best_average_loss = average_loss
        best_uid = uid
        best_uid_run = run
        if average_loss < best_average_loss:
            best_average_loss = average_loss
            best_uid = uid
            best_uid_run = run
        elif average_loss == best_average_loss:
            if run.created_at <= best_uid_run.created_at:
                best_average_loss = average_loss
                best_uid = uid
                best_uid_run = run

        if best_uid != None:
            weights = torch.zeros_like( metagraph.S )
            weights[best_uid] = 1
        else:
            weights = torch.ones_like( metagraph.S )

        subtensor.set_weights(
            netuid = pretrain.NETUID,
            wallet = wallet,
            uids = metagraph.uids,
            weights = weights,
            wait_for_inclusion=False,
        )
        metagraph = subtensor.metagraph( pretrain.NETUID )
    


