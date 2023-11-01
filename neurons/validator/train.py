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

# import what I need
import random
import torch
import typing
import pretrain
import bittensor as bt

# === Train loop ===
async def train_loop( self ):
    while True:
        loss = await train( self )
        if loss < self.best_average_loss:
            self.best_average_loss = loss
            self.best_model_state = self.model.state_dict()
            bt.logging.success(f"New best average loss: {self.best_average_loss}")

# === Eval ===
async def train(self):
    # Get Random page. 
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
        inputs = batch.to( self.model.device )
        outputs = self.model( inputs, labels=inputs )
        outputs.loss.backward()
        average_loss += outputs.loss.detach().item()
        n_batches += 1
        bt.logging.success( f'Acc: step: {i} loss: {outputs.loss}' )
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.empty_cache()

    # Return average loss on batch.
    return average_loss / n_batches
