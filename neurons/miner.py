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
import random
from argparse import ArgumentParser
from functools import cached_property
from itertools import islice
from typing import Iterator

import bittensor as bt
import torch
from transformers.models.auto.auto_factory import _BaseAutoModelClass

import pretrain as pt


class NotRegisteredError(Exception):
    pass


class Miner:
    config: bt.config
    wallet: bt.wallet
    subtensor: bt.subtensor
    metagraph: bt.metagraph

    @classmethod
    def get_argument_parser(cls) -> ArgumentParser:
        """
        Set up and parse the command-line arguments to configure the system.

        The configuration is responsible for setting up the environment including
        the model path, device to use, and the bittensor wallet and logging configurations.
        """

        # Initialize an argument parser
        parser = ArgumentParser()

        # Do not upload model to huggingface
        parser.add_argument( '--offline', action='store_true', help='Do not upload model to HuggingFace' )

        # Add model_path argument which allows the user to specify the path of the model
        parser.add_argument("--huggingface_repo_name", type=str, default="", help="Please clarify the huggingface repo name (your huggingface space)/(model_name) e.g. James/model1")

        # Add model_path argument which allows the user to specify the path of the model
        parser.add_argument("--huggingface_api_token", type=str, default="", help="Please only give api token")

        # Save the model before running any additional steps
        parser.add_argument( '--early_publish', action='store_true', help='Early save the model to HuggningFace' )

        # Add model_path argument which allows the user to specify the path of the model
        parser.add_argument("--model_path", type=str, required=False, help="Override model path")

        # Add device argument which defaults to 'cuda' if available, else 'cpu'
        parser.add_argument("--device", type=str, choices=("cuda", "cpu"), default="cuda" if torch.cuda.is_available() else "cpu", help="Device name.")

        # Add device argument which defaults to 'cuda' if available, else 'cpu'
        parser.add_argument("--load_best", action='store_true', help='If set, the miner loads the best model from wandb to train off.' )

        # Add device argument which defaults to 'cuda' if available, else 'cpu'
        parser.add_argument("--load_uid", type=int, default=None, help='If passed loads the model under the specified uid.' )

        # Add device argument which defaults to 'cuda' if available, else 'cpu'
        parser.add_argument("--load_disk",action='store_true', help='If set, loads the model from disk' )

        # Set the number of epochs
        parser.add_argument("--num_epochs", type=int, default=-1, help="Number of training epochs (-1 is infinite)")

        # Training lr.
        parser.add_argument("--lr", type=float, default=0.00001, help="Learning rate.")

        # Training batch size
        parser.add_argument("--bs", type=int, default=pt.batch_size, help="Batch size")

        # Training sequence length
        parser.add_argument("--sl", type=int, default=pt.sequence_length, help="Sequence length")

        # Training accumulation steps per step.
        parser.add_argument("--accumulation_steps", type=int, default=5, help="The number of training accumulation steps")

        # Set the number of pages trained per epoch
        parser.add_argument("--pages_per_epoch", type=int, default=10, help="Number of pages trained on per epoch")

        # Include bittensor-specific arguments
        bt.wallet.add_args(parser)
        bt.subtensor.add_args(parser)
        bt.logging.add_args(parser)

        return parser

    def __init__(self) -> None:
        parser = self.get_argument_parser()
        self.config = config = bt.config(parser)

        bt.logging(config=config)
        bt.logging.info(config)

        self.wallet = bt.wallet(config=config)
        self.subtensor = bt.subtensor(config=config)
        self.metagraph = self.subtensor.metagraph(pt.NETUID)

    def upload_model(self, model: _BaseAutoModelClass) -> None:
        """
        Upload the model to HuggingFace.
        """
        if self.config.offline:
            bt.logging.info('Skipping model upload because offline flag is set')
            return

        if not pt.mining.is_valid_size(model):
            raise ValueError('Model size is not valid, please check your setting.')

        if not (huggingface_repo_name := self.config.huggingface_repo_name):
            raise ValueError('Please specify the huggingface repo name')

        _push_flag = pt.mining.push(
            model,
            huggingface_repo_name,
            self.config.huggingface_api_token,
        )

        data = huggingface_repo_name
        bt.extrinsics.serving.publish_metadata(
            self.subtensor,
            self.wallet,
            netuid=pt.NETUID,
            type=f"Raw{len(data)}",
            data=data.encode(),
        )

        bt.logging.success(f'Saving model to path: {pt.mining.path(self.wallet)}.')
        pt.mining.save(self.wallet, huggingface_repo_name)

    def load_model(self) -> _BaseAutoModelClass:
        # Initialize the model based on the best on the network.
        if self.config.load_best:
            # Get the best UID be incentive and load it.
            best_uid = pt.graph.best_uid(self.metagraph)
            repo_name = self.subtensor.get_commitment(pt.NETUID, best_uid)
            model = pt.mining.load_from_hf(self.wallet, repo_name)
            bt.logging.success(f'Loaded model with best uid: {best_uid}')
            return model

        # Initialize the model based on a passed uid.
        if (load_uid := self.config.load_uid) is not None:
            # Sync the state from the passed uid.
            repo_name = self.subtensor.get_commitment(pt.NETUID, load_uid)
            model = pt.mining.load_from_hf(self.wallet, repo_name)
            bt.logging.success(f'Loaded model from uid: {load_uid}')
            return model

        # Initialize the model from scratch.
        if self.config.load_disk:
            model = pt.mining.load(self.wallet)
            bt.logging.success('Loaded model from disk')
            return model

        model = pt.model.get_model()
        bt.logging.success('Created model from scratch')
        return model

    @cached_property
    def ss58_address(self) -> str:
        return self.wallet.hotkey.ss58_address

    def check_registration(self) -> None:
        if self.config.offline:
            return

        if self.ss58_address not in self.metagraph.hotkeys:
            raise NotRegisteredError(
                f"You are not registered. \n"
                f"Use: \n`btcli s register --netuid {pt.NETUID}` to register via burn \n "
                f"or btcli s pow_register --netuid {pt.NETUID} to register with a proof of work"
            )

        uid = self.metagraph.hotkeys.index(self.ss58_address)
        bt.logging.success(f'You are registered with address: {self.ss58_address} and uid: {uid}')

    def iter_epochs(self, model: _BaseAutoModelClass) -> Iterator[float]:  # returns avg loss for each epoch
        n_acc_steps = 0
        accumulation_steps = self.config.accumulation_steps
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr, weight_decay=0.01)

        while True:

            # Initialize loss accumulator for the epoch
            epoch_loss = 0.0

            # Prepare the data loader with random pages for each epoch
            bt.logging.info( f"Loading {self.config.pages_per_epoch} pages for training this epoch" )
            random_pages = [
                random.randint(1, pt.dataset.SubsetFalconLoader.max_pages)
                for _ in range(self.config.pages_per_epoch)
            ]
            loader = pt.dataset.SubsetFalconLoader(
                batch_size = self.config.bs,
                sequence_length = self.config.sl,
                pages = random_pages,
            )

            # Enumerate over the data loader
            n_batches = 0
            optimizer.zero_grad()  # Initialize gradients to zero

            for i, batch in enumerate(loader):
                # Move the input batch to the device
                inputs = batch.to(model.device)

                # Forward pass: compute the model output and loss
                outputs = model(inputs, labels=inputs)

                loss = outputs.loss / accumulation_steps  # Scale loss
                loss.backward()  # Accumulate gradients

                if (i + 1) % accumulation_steps == 0:
                    n_acc_steps += 1
                    optimizer.step()  # Perform a single optimization step
                    optimizer.zero_grad()  # Clear gradients
                    bt.logging.success(f'Step: {n_acc_steps} loss: {outputs.loss.detach().item()}')

                torch.cuda.empty_cache()

                # Log the loss for the current step
                n_batches += 1
                epoch_loss += outputs.loss.detach().item()

            # Calculate the average loss for the epoch
            yield epoch_loss / n_batches

    def run(self) -> None:
        self.check_registration()

        model = self.load_model()
        model.train()
        model.to(self.config.device)

        if self.config.early_publish:
            self.upload_model(model)

        # Start the training loop
        best_avg_loss = math.inf
        avg_losses = self.iter_epochs(model)
        if self.config.num_epochs != -1:
            avg_losses = islice(avg_losses, self.config.num_epochs)

        for epoch_step, avg_loss in enumerate(avg_losses):
            # Log the average loss for the epoch
            bt.logging.success(f'Epoch: {epoch_step} average loss: {avg_loss}')

            # Check if the average loss of this epoch is the best we've seen so far
            if avg_loss < best_avg_loss * ( 1 - pt.timestamp_epsilon ):
                best_avg_loss = avg_loss  # Update the best average loss
                bt.logging.success(f'New best average loss: {best_avg_loss}')
                self.upload_model(model)


if __name__ == "__main__":
    Miner().run()
