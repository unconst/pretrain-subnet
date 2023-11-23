
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
import pretrain
from transformers import GPT2Config, GPT2LMHeadModel

config = GPT2Config()

def get_model():
    config = GPT2Config(
        n_head = 10,
        n_layer = 12,
        n_embd = 760,
    )
    return GPT2LMHeadModel(config)


def get_model_for_uid( uid:int, device:str) -> torch.nn.Module:
    model = pretrain.model.get_model()
    model_meta = pretrain.utils.load_metadata_for_uid( uid )
    model_weights = torch.load( model_meta['model_path'], map_location=torch.device(device))
    model.load_state_dict( model_weights )
    return model


