# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# TODO(developer): Set your name
# Copyright © 2023 <your name>

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

__version__ = "0.0.0"
version_split = __version__.split(".")
__spec_version__ = (
    (1000 * int(version_split[0]))
    + (10 * int(version_split[1]))
    + (1 * int(version_split[2]))
)
NETUID = 9


# Define model configuration.
from transformers import GPT2Config, GPT2Tokenizer
MODEL_CONFIG = GPT2Config(
    n_layer = 12, 
    n_head = 12,
)
TOKENIZER = GPT2Tokenizer.from_pretrained('gpt2')
TOKENIZER.pad_token = TOKENIZER.eos_token

# Define Dataloader.
# Using the Falcon refined web huggingface api directly. We cycle through all 968000015 rows at random.
# The iterator of this class returns sequences of length `sequence_length` which are randomly pulled from
# the Falcon refined web corpus. 
import torch
import random
import requests
from torch.utils.data import DataLoader, IterableDataset

class FalconDataset(IterableDataset):
    def __init__(self, tokenizer, sequence_length):
        self.buffer = []
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.num_rows_total = 968000015
        self.num_rows_per_page = 100
        self.base_url = "https://datasets-server.huggingface.co/rows"
        self.params = {
            "dataset": "tiiuae/falcon-refinedweb",
            "config": "default",
            "split": "train"
        }
        
    def get_random_data(self):
        # Gets a random page of data from the URL and returns the json rows.
        random_offset = random.randint(0, self.num_rows_total - self.num_rows_per_page)
        self.params["offset"] = random_offset
        self.params["limit"] = self.num_rows_per_page
        response = requests.get(self.base_url, params=self.params)
        if response.status_code == 200:
            return response.json()["rows"]
        else:
            return []

    def __iter__(self):
        while True:
            # If buffer is empty, fetch a new random data page
            if not self.buffer:
                data = self.get_random_data()
                for row in data:
                    content = row["row"]["content"]
                    self.buffer += self.tokenizer(content)["input_ids"]
                    self.buffer += [self.tokenizer.eos_token_id]
            # Yield data in sequence length chunks until buffer is exhausted
            if len(self.buffer) > self.sequence_length:
                yield torch.tensor(self.buffer[:self.sequence_length])
                self.buffer = self.buffer[self.sequence_length:]
            else:
                # If there's not enough data in the buffer for another sequence,
                # we will just clear the buffer and fetch new data in the next iteration.
                self.buffer = []


