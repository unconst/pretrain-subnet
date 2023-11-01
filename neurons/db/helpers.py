
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
import typing
import pretrain
import bittensor as bt

def init_wandb( self: object, type: str, uid: int, reinit=False ):
    """Starts a new wandb run."""
    if self.config.wandb.on:
        tags = [ type, f'uid:{uid}', self.wallet.hotkey.ss58_address, pretrain.__version__, str(pretrain.__spec_version__), f'netuid_{pretrain.NETUID}']
        return wandb.init(
            anonymous = "allow",
            reinit = reinit,
            project = self.config.wandb.project_name,
            entity = self.config.wandb.entity,
            config = self.config,
            mode = "offline" if self.config.wandb.offline else "online",
            dir = self.config.full_path,
            tags=tags,
        )
    else:
        return None
