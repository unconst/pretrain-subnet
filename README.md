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

Bittensor subnet 9 is currently designed to reward miners for producing producing pretrained models on the Falcon Refined Web dataset. It acts like a continuous benchmark where miners are paid out for attaining the best losses on randomly sampled pages of that dataset. 
The reward mechanism works as follows.

    1. Miner train and periodically host their model weights on a wandb account which linked to their miner through the neurons/miner.py code. 
    2. Validators periodically check and pull the latest models hosted on these miners which can be updated continuously by writing the models to ~/.bittensor/miners/<your cold wallet>/<your hot wallet>/netuid9/model.pth
    3. Validators run a continuous eval on the models hosted by miners performing the validation system outlined in neurons/validator.py setting weights every epoch based on the performance of each miner on the Falcon dataset.

#### Validation

Miners are evaluated based on the number of times their loss on a random batch duing a 360 block epoch are lower than all other miners. 
To perform well miners must attain the lowest loss on the largest number of random batches sampled from the 900M page, 3T token dataset Falcon Refined Wed.
```python
    while True:
        wins = {} # Count of wins per batch per miner

        # Run continous scoring until the epoch is over.
        while epoch_not_over( block )

            # Fetch random sample of batches to evaluate models on
            batches = get_random_sample_of_batches_from_falcon()
            
            # Fetch and or update models during this step.
            Models = get_and_update_models_from_miners()

            # Compute losses for each batch on subset and count wins per miner
            for batch in Batches:

                # Find miner with lowest loss on the batch.
                for miner_uid, model in enumerate( Models ):
                    loss = get_loss_for_model_on_batch( model, batch )
                    if loss < best_loss:
                        best_uid = miner_uid
                        best_loss = loss

                # Increment the number of wins for the miner with the lowest loss on this subnet.
                wins[ best_uid ] += 1

        # End epoch.
        # Weights are computed based on the ratio of wins a model attains during the epoch.
        weights = zeros()
        for miner_uid in wins.keys()
            weights = wins[miner_uid] / sum(wins.values())

        # Set weights on the chain.
        set_weights( weight )
```

---

## Mining

Before mining make sure you have python3.8. Then install this repository.
```bash
git clone https://github.com/unconst/pretrain-subnet.git
cd pretrain-subnet
python -m pip install -e . 
```

Miners must attain a wandb account from [wandb](https://wandb.ai/home) and attain their wandb api key.
Follow the instructions [here](https://docs.wandb.ai/quickstart)

```bash
wandb init
```

Once your wandb is installed. Train your model. Note, the training script is simply a mock training script, we recommend you ammend the training script at later date.
Or copy the weights from other miners, subnet 9 is pro model sharing as a means of distributing contiously better models to participants. 
```bash
python neurons/train.py --wallet.name ... --wallet.hotkey ... 
```

The mode trained by the running the above training script will be written to ```~/.bittensor/miners/<your cold>/<your hot>/netuid9/miner/model.pth```. Your miner will find this
model and periodically advertise if it changes over time. To run a miner over the model save here run the following command.

```bash
python neurons/miner.py --wallet.name ... --wallet.hotkey ...
```



---

## Wallets


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
