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
import typing
import requests
import bittensor as bt
from torch.utils.data import IterableDataset
from transformers import GPT2Tokenizer

model_name = 'distilgpt2'

class SubsetFalconLoader(IterableDataset):
    
    max_pages: int = 968000015

    def __init__(self, batch_size, sequence_length, pages: typing.List[int] ):
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_rows_per_page = 100
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.base_url = "https://datasets-server.huggingface.co/rows"
        self.params = {
            "dataset": "tiiuae/falcon-refinedweb",
            "config": "default",
            "split": "train"
        }
        self.pages = pages
        self.buffer = []
        for page in self.pages:
            self.params["offset"] = page
            self.params["limit"] = self.num_rows_per_page
            response = requests.get(self.base_url, params=self.params)
            for row in response.json()["rows"]:
                content = row["row"]["content"]
                self.buffer += self.tokenizer(content)["input_ids"]
                self.buffer += [self.tokenizer.eos_token_id]
            
    def __iter__(self):
        while len(self.buffer) >= self.sequence_length * self.batch_size:
            batch = []
            for _ in range(self.batch_size):
                batch.append(torch.tensor(self.buffer[:self.sequence_length]))
                self.buffer = self.buffer[self.sequence_length:]
            yield torch.stack(batch)
            
    def __next__(self):
        batch = []
        for _ in range(self.batch_size):
            batch.append(torch.tensor( self.buffer[:self.sequence_length] ))
            self.buffer = self.buffer[self.sequence_length:]
        return torch.stack(batch)
            