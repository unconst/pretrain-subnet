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
    parser.add_argument( "--model_path", type = str, help="Run name.", required=True )
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
loss_dict = {}

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
    metagraph = subtensor.metagraph( pretrain.NETUID )
    uids = metagraph.uids
    for uid in uids:
        loss_dict[uid] = {}
        axon = metagraph.axon[uid]

        run_name = dendrite.query( axon, pretrain.GetRun() ).run_name
        run = api.run(f"opentensor-dev/openpretraining/{run_name}")
        loss_dict["uid"]["run_name"] = run
        loss_dict["uid"]["timestamp"] = run.created_at

        hotkey = run.config.get('hotkey')
        loss_dict["uid"]["hotkey"] = hotkey
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
        average_loss = None
        for i, batch in enumerate( loader ):
            try:
                average_loss = 0
                inputs = batch.to( model.device )
                outputs = model( inputs, labels=inputs )
                outputs.loss.backward()
                average_loss += outputs.loss.detach().item()
                torch.cuda.empty_cache()
                bt.logging.success( f'Acc: step: {i} loss: {outputs.loss}' )
                if average_loss < loss_dict["uid"]["loss"]:
                    loss_dict["uid"]["loss"] = average_loss
            except Exception as e:
                bt.logging.exception(f"Error in loss calc of {uid} \n {e}")


    # Get best average loss and best uid
    # Best uid if tie on loss is based on timestamp of run upload
    try:
        best_average_loss = min(loss_dict[uid]['loss'] for uid in loss_dict if loss_dict[uid]['loss'] is not None)
        uids_with_best_average_loss = [uid for uid, values in loss_dict.items() if values['loss'] == best_average_loss]
        
        if len(uids_with_best_average_loss) > 1:
            # Assuming each uid has a 'timestamp' key
            best_uid = min(uids_with_best_average_loss, key=lambda uid: loss_dict[uid]['timestamp'])
        
        elif len(uids_with_best_average_loss) == 1:
            best_uid = uids_with_best_average_loss[0]
        
        else:
            best_uid = None

    except ValueError:
        best_uid = None
        bt.logging.error("All loss None! Setting all scores to 0.")

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
    


