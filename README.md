<div align="center">

# **Bittensor Training Subnet** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## Bittensor Incentivized Pretraining<!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---

# Introduction

Bittensor subnet 9 is currently designed to reward miners for producing gradients which can be used to train LLMs. In its first iteration Subnet 9 only produces gradients for a GPT2 model and uses the Falcon Refined Web as the base dataset. Clients of this network can broadcast requests to miners to produce gradients and aggregate them to train their local model. Subnet 9 is still in development phase. 

---

## Installation

Make sure you have python3.8 or above before continuing.
```bash
git clone https://github.com/unconst/pretrain-subnet.git
cd pretrain-subnet
python -m pip install -e . 
```

---

## Preable

To mine or validate first create your wallet with cold and hotkey pair, and register it to subnet 9.
For instance, creating 1 coldkey wallet with 3 hotkeys for a validator and two miners.

```bash
# Create your wallet and hotkeys.
btcli w new_coldkey --wallet.name my_wallet
btcli w new_hotkey --wallet.name my_wallet --wallet.hotkey miner1
btcli w new_hotkey --wallet.name my_wallet --wallet.hotkey miner2
btcli w new_hotkey --wallet.name my_wallet --wallet.hotkey validator

# Register your hotkeys to the main chain.
# NOTE: this costs TAO.
btcli s recycle_register --wallet.name my_wallet --wallet.hotkey miner1
btcli s recycle_register --wallet.name my_wallet --wallet.hotkey miner1
btcli s recycle_register --wallet.name my_wallet --wallet.hotkey validator
```
---

## Local Subtensor

Your node will run better if you are connecting to a local Bittensor chain entrypoint node rather than using Opentensor's. 
We recommend running a local node as follows and passing the ```--subtensor.network local``` flag to your running miners/validators. 
To install and run a local subtensor node follow the commands below with Docker and Docker-Compose previously installed.
```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
docker compose up --detach
```


---

## Running

Miners produce gradients using their local machines. You can run a miner like so.
```bash
# Miner script.
#   python neurons/miner.py
#
# Miner wallet name
#    --wallet.name i.e. miner
#
# Miner hotkey must be distinct per miner
#    --wallet.hotkey i.e. my_hotkey
#
# Select a device (different for each miner), if you dont have a GPU pass 'cpu'  
#    --device cuda:1  
#
# Each miner must have a separate port here (also different for each miner)
#    --axon.port 8091 
#
# To use wandb pass 
#    --wandb.on
#
# To run connect to your local subtensor entrypoint
#   --subtensor.network local 

# Run first miner
python neurons/miner/run.py --wallet.name my_wallet --wallet.hotkey miner1 --device cuda:1 --logging.debug --axon.port 9091 --wandb.on --subtensor.network local

# Run your second miner
python neurons/miner/run.py --wallet.name my_wallet --wallet.hotkey miner2 --device cuda:2 --logging.debug --axon.port 9092 --wandb.on --subtensor.network local
```

Validators train a GPT2 model over the network and validate the gradients produced by the miners.
You can run your validator/trainer like so.
```bash
# Validator name:
#   python neurons/validator.py
#
# The validator wallet name:
#    --wallet.name i.e. validator 
#
# The validator hotkey name:
#    --wallet.hotkey i.e. default 
#
# The validator device, different that miners:
#    --device i.e. cuda:0
# 
# To use wandb pass 
#    --wandb.on
#
# To run connect to your local subtensor entrypoint
#   --subtensor.network local 

# Run the validator
python neurons/validator/run.py --wallet.name my_wallet --wallet.hotkey validator --logging.debug --device cuda:3 --wandb.on --subtensor.network local
```

---

# Auto-update PM2 + CRON

```bash
echo '* * * * * git -C <path to pretrain-subnet repo> pull' >> /etc/crontab
pm2 start neurons/validator/run.py --name sn9_validator --interpreter python3 --watch -- --wallet.name my_wallet --wallet.hotkey validator --logging.debug --device cuda:1  --subtensor.network local
pm2 start neurons/miner/run.py --name sn9_miner_1 --interpreter python3 --watch -- --wallet.name my_wallet --wallet.hotkey miner1 --logging.debug --device cuda:2  --subtensor.network local
pm2 start neurons/miner/run.py --name sn9_miner_2 --interpreter python3 --watch -- --wallet.name my_wallet --wallet.hotkey miner2 --logging.debug --device cuda:3  --subtensor.network local
```

---

## License
This repository is licensed under the MIT License.
```text
# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

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
```
