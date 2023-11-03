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
import torch
import argparse
import pretrain
import bittensor as bt

# === Config ===
def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ss58", type=str, help="ss58 of mining hotkey", default=None)
    known_args, _ = parser.parse_known_args()
    ss58 = known_args.ss58
    default_model_path = f'~/pretrain-subnet/neurons/pretraining_model/{ss58}' if ss58 else '~/pretrain-subnet/neurons/pretraining_model/default'

    # Now add the model_path argument with the proper default
    parser.add_argument("--model_path", type=str, help="Run name.", default=default_model_path)
    parser.add_argument( "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")
    bt.logging.add_args( parser )
    config = bt.config(parser)
    return config
config = get_config()

model_path = os.path.expanduser(config.model_path)

model = pretrain.model.get_model()
model.zero_grad()
model.train()
model.to( config.device )

optimizer = torch.optim.AdamW( model.parameters(), lr=0.000001, weight_decay=0.01 )
loader = pretrain.dataset.SubsetFalconLoader( batch_size=3, sequence_length=512, pages=[1, 2, 3, 4, 5] )

num_epochs = 5 
for epoch in range(num_epochs):
    for i, batch in enumerate(loader):
        inputs = batch.to( model.device )
        outputs = model( inputs, labels=inputs )
        outputs.loss.backward()
        torch.cuda.empty_cache()
        optimizer.step()
        bt.logging.success( f'Acc: step: {i} loss: {outputs.loss}' )

bt.logging.success( f'Saving model to {model_path}' )
torch.save( model.state_dict(), model_path )