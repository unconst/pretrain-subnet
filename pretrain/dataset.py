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
import random
import requests
import threading
import bittensor as bt
from torch.utils.data import DataLoader, IterableDataset
from transformers import GPT2LMHeadModel, GPT2Config, GPT2Tokenizer

# --- Dataloader
# Using the Falcon refined web huggingface api directly. We cycle through all 968000015 rows at random.
# The iterator of this class returns sequences of length `sequence_length` which are randomly pulled from
# the Falcon refined web corpus. 
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



class ThreadedFalconDataset(IterableDataset):
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
        self.buffer_lock = threading.Lock()
        self.buffer_capacity = 2 * sequence_length  # assuming you want the capacity to be twice the sequence length
        self.running = True
        self.buffer_fill_thread = threading.Thread(target=self.fill_buffer)
        self.buffer_fill_thread.start()

    def get_random_data(self):
        random_offset = random.randint(0, self.num_rows_total - self.num_rows_per_page)
        self.params["offset"] = random_offset
        self.params["limit"] = self.num_rows_per_page
        response = requests.get(self.base_url, params=self.params)
        if response.status_code == 200:
            return response.json()["rows"]
        else:
            return []

    def fill_buffer(self):
        while self.running:
            if len(self.buffer) < 0.5 * self.buffer_capacity:
                data = self.get_random_data()
                for row in data:
                    content = row["row"]["content"]
                    with self.buffer_lock:
                        self.buffer += self.tokenizer(content)["input_ids"]
                        self.buffer += [self.tokenizer.eos_token_id]

    def __iter__(self):
        while True:
            if len(self.buffer) > self.sequence_length:
                with self.buffer_lock:
                    yield torch.tensor(self.buffer[:self.sequence_length])
                    self.buffer = self.buffer[self.sequence_length:]
            else:
                # Optional sleep to reduce aggressive polling
                time.sleep(0.1)

    def __del__(self):
        self.running = False
        self.buffer_fill_thread.join()  # Wait for the thread to finish


# --- Get single thread dataloader.
def get_dataloader(bs, sq):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset = FalconDataset(tokenizer=tokenizer, sequence_length=sq)
    dataloader = iter(DataLoader(dataset, batch_size=bs, num_workers=1))
    return dataloader

# --- Get threaded dataloader.
def get_threaded_dataloader( bs, sq ):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    dataset = ThreadedFalconDataset( tokenizer = tokenizer, sequence_length = sq )
    dataloader = iter( DataLoader( dataset, batch_size = bs, num_workers = 1 ) )
    return dataloader