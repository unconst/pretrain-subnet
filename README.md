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

Bittensor subnet 9 rewards miners (engineers etc) for producing pretrained Foundation-Models on the Falcon Refined Web dataset. It acts like a continuous benchmark, whereby miners are paid out for attaining the best losses on randomly sampled pages of that dataset given a consistent model architecture and eval dataset. The reward mechanism works as follows:

    1. Miners train and periodically host trained model weights linked to their miner key as exampled by the code in neurons/miner.py.
    2. Validators run a continuous eval on the hosted models, performing the validation system outlined in neurons/validator.py and setting weights to the chain based on the performance of each miner on the Falcon dataset.

---

## Getting started

This repo's main conversation is carried out in the Bittensor [Discord](https://discord.gg/bittensor) under 'pretraining' which is a great place to ask questions and get real time feedback. You can view the best miners on the network by using a tool like [taostats](https://taostats.io/) to see all 256 participant slots under subnet 9, and you can view all current wandb runs for miners and validators [here](https://wandb.ai/opentensor-dev/pretraining-subnet). To understand how Bittensor works under the hood please refer to our documentation on Yuma Consensus, and or read the original [paper](https://bittensor.com/whitepaper).

---

## Installing

Before beginning to mine or validate on this subnet you will need to install `pretrain` and `bittensor`. You will need at least python3.8 to install this repo we recommend using a package manager like anaconda.
```bash
git clone https://github.com/unconst/pretrain-subnet.git
cd pretrain-subnet
python -m pip install -e . 
```

Once installed correctly you can import these packages in python.
```python
import bittensor as bt
import pretrain as pt
```

## Validation

Miners are evaluated based on the number of times their loss on a random batch from Falcon is lower than other miners. To perform well, miners must attain the lowest loss on the largest number of random batches sampled from the 900M page, 3T token dataset, Falcon Refined Wed. All models are open and accessible via a wandb [project](https://wandb.ai/opentensor-dev/pretraining-subnet) and this repo contains tools for downloading them,
serving them to your miner, as well as getting data from validators about which models perform best on which pages of the Falcon Dataset. Finding the best model and delta at the earliest timestamp 
ensures the most incentive.

```python
    while True:
        wins = {} # Count of wins per batch per miner

        # Run continous scoring until the epoch is over.
        while epoch_not_over( block )

            # Fetch random sample of batches to evaluate models on
            batches = get_random_sample_of_batches_from_falcon()
            
            # Fetch and or update models during this step.
            models = get_and_update_models_from_miners()

            # Compute losses for each batch and each model
            model_losses = {}
            for model in models:
                for batch in batches:
                    loss = get_loss_for_model_on_batch( model, batch )
                    model_losses[ model ].append( loss )

            # Compute wins for models.
            model_wins = {}
            for model_a in models:
                for model_b in models:
                    for i in len( batches )
                        # Determine if better model loss with timestamp boosting.
                        if iswin( model_losses[ model_a ][ i ], model_losses[ model_b ][ i ], timestamp_a, timestamp_b ):
                            model_wins[ model_a ] += 1
                            
        # End epoch.
        # Weights are computed based on the ratio of wins a model attains during the epoch.
        weights = zeros()
        for model_i in models:
            weights += model_wins[ model_i ] / sum( model_wins.values() )

        # Set weights on the chain.
        set_weights( weight )
```

The behaviour of the `iswin( loss_a, loss_b, timestamp_a, timestamp_b)` function is integral to the way a which this incentive function works since it skews the win function 
to reward models which have been hosted earlier on wandb. Specifically, a newer model or delta is only better than another if it is more than `epsilon` percent lower accoring to
the following function.
```python

def iswin( loss_a, loss_b, timestamp_a, timestamp_b, epsilon ):
    loss_a = (1 - epsilon) * loss_a if timestamp_a < timestamp_b else loss_a
    loss_b = (1 - epsilon) * loss_b if timestamp_b < timestamp_a else loss_b
    if loss_a < loss_b: return True
    else: return False
```

It is important to note that this affects the game theoretics of the incentive landscape since miners should only update their delta or model (thus updating their timestamp to a newer date) if they
have achieved an `epsilon` better loss on average on the Falcon Refined Web dataset. This undermines the obvious optimal strategy for miners to copy the publicly available models and deltas
of other miners. They **can** and should copy other miners, but they will always obtain fewer wins compared to them until they also decrease their loss by `epsilon`. The effect is to drive miners to 
continually produce better models.

--- 
## Wandb

Miner and validators make heavy use of weights and biases in order to share model state and validation information. Both miners and validators must attain
a wandb account from [wandb](https://wandb.ai/home) along with their wandb api key which can be found by following the instructions [here](https://docs.wandb.ai/quickstart).

---

## Subtensor

Your node will run better if you are connecting to a local Bittensor chain entrypoint node rather than using Opentensor's. 
We recommend running a local node as follows and passing the ```--subtensor.network local``` flag to your running miners/validators. 
To install and run a local subtensor node follow the commands below with Docker and Docker-Compose previously installed.
```bash
git clone https://github.com/opentensor/subtensor.git
cd subtensor
docker compose up --detach
```

---

## Registration

Miners + validator require a Bittensor coldkey and hotkey pair registered to netuid 9 before they can participate in the network.
To create a wallet for either your validator or miner run the following command in your terminal. Make sure to save the mnemonic for
both keys and store them in a safe place.
```bash
# to create your miner/validator cold + hotkey keys.
btcli w create --wallet.name ... --wallet.hotkey ... 
btcli w list # to view your created keys.
```

Registering a miner or a validator on subnet 9 requires the participant `recycle` TAO to pay for entrance. To register your key run the 
following command. If you don't have any TAO message const [t,t] on Discord for a faucet to try things out, please don't scam me.
```bash
# register your cold and associated hotkey to netuid 9
btcli s register --wallet.name ... --wallet.hotkey ... --netuid 0 
```

---

## Mining

The mining script can be run in multiple ways. In all cases, it uploads a model to wandb which will be evaluated by validators. 

By default, the miner trains from scratch and posts the model periodically as its loss decreases on Falcon.
```bash
python neurons/miner.py --wallet.name ... --wallet.hotkey ... --num_epochs 10 --pages_per_epoch 5
```

Alternatively, you can scrape a model from an already running miner on the network by passing its uid. This starts the training process from the checkpoint of another 
miner.
```bash
python neurons/miner.py --wallet.name ... --wallet.hotkey ... --num_epochs 10 --pages_per_epoch 5 --load_uid ...
```

The miner can automatically search the miner runs directly for the participant with the best score and use that as the main checkpoint. The pretraining
subnet is PRO model sharing, so we recommend miners scrap other participants models and often.
```bash
python neurons/miner.py --wallet.name ... --wallet.hotkey ... --num_epochs 10 --pages_per_epoch 5 --load_best
```

Passing the ```--device``` option allows you to select which GPU to run on. You can also use ```--continue_id``` to continue from a training run you have already started.
The model you train will be hosted on wandb. You can always view this model and others by visiting https://wandb.ai/opentensor-dev/pretrain.WANDB_PROJECT/runs/ where all participant 
model are shared. 

---

## Validating

Validators can be run as follows. Pass you wallet hotkey and coldkey to the script. Note validation required you have a working GPU.
In version release/2.0.1 you need a GPU with atleast 20GB of RAM. 

```bash
python neurons/validator.py 
    --wallet.name YOUR_WALLET_NAME
    --wallet.hotkey YOUR_WALLET_HOTKEY 
    --device YOUR_CUDA DEVICE
    --wandb.on 
```
---

# Auto-update PM2 + CRON

```bash
echo '* * * * * git -C <path to pretrain-subnet repo> pull' >> /etc/crontab
pm2 start neurons/validator.py --name sn9_validator --interpreter python3 --watch -- --wallet.name my_wallet ...
pm2 start neurons/miner.py --name sn9_miner_1 --interpreter python3 --watch -- --wallet.name my_wallet ...
pm2 start neurons/miner.py --name sn9_miner_2 --interpreter python3 --watch -- --wallet.name my_wallet ...
```

---

## Bittensor API

The Bittensor repository comes with tooling for interfacing with the Bittensor ecosystem, creating wallets, loading state from the chain, and registering miners and validators into the mechanism. Before continuing make sure you are familiar with the main concepts of Bittensor.

```python
import bittensor as bt

# Accesses the incentive mechanism state of the pretraining 'subnet'. A subnet is a self contained consensus engine through which miners and validators agree on value creation and through which 
# TAO (bittensor's value holding token) is distributed. The metagraph is a syncable torh object which contains the state of this consensus engine at a particular block. Here we are pulling the state
# for subnet 9, which is the subnet network id, associated with this pretraining mechanism.
metagraph = bt.metagraph( 9 )
print( metagraph )

# Participants in a Bittensor consensus mechanism are defined by their wallet which contains two cryptographic keypairs, a coldkey and a hotkey. The hotkey has low security and is used to sign messages from a running miner
# while the coldkey is encrypted at all times and used to store and move TAO. The following code snippet creates a wallet on your machine with the specified coldkey name and hotkey name.
wallet = bt.wallet( name = 'cold', hotkey = 'hot' ).create_if_non_existent()
print( wallet )
```

## Pretrain API

The pretraining repo is for the easily constructing participants (miners / validators) within subnet 9 and loading and evaluating the state of the network. For instance, as a means of pushing models you have trained, or for attaining other participants' models as checkpoints for you.

Creating miners.
```python
import bittensor as bt
import pretrain as pt

# Create a mining wallet.
wallet = bt.wallet().create_if_non_existent()

# Output your mining directory.
print (f'''Wallet: {wallet}
    path: {pt.mining.path( wallet )}
    model_path: {pt.mining.model_path( wallet )}
    runidpath: {pt.mining.runidpath( wallet )}
''')

# Init or reinit the wandb run associtated with this wallet.
wandb_run = pt.mining.init( wallet )

# Push a new model to your wandb run.
pt.mining.push( pt.model.get_model(), wallet, wandb_run )
```

The pretrain package also contains the following commands for pulling state from the network and performing validation. 

```python
import pretrain as pt
device = 'cuda'

# Pulls/Downloads model information and stores it onto your harddrive under `~/.bittensor/miners/netuid9/models/231/*`
pt.graph.sync( 231 )
pt.graph.sync( 200 )

# Print information about the recently synced uid.
print (f'''UID 231:
    timestamp: {pt.graph.timestamp( 231 )}
    run: {pt.graph.run( 231 )} 
    runid: {pt.graph.runid( 231 )}
    version: {pt.graph.version( 231 )}
    model_path: {pt.graph.model_path( 231 )}
    hotkey: {pt.graph.hotkey( 231 )}
    last_update: {pt.graph.last_update( 231 )}
''')

# Load downloaded model from harddrive to device.
model_231 = pt.graph.model( 231, device = device )
model_200 = pt.graph.model( 200, device = device )

# Attains batches from the Falcon Dataset based on pages 101
batches = list(pretrain.dataset.SubsetFalconLoader( batch_size = 2, sequence_length = 1024, pages = [ 101 ] ) )

# Evaluate the models on these batches.
losses_231 = pretrain.validation.compute_losses( model_231, batches, device = device )
losses_200 = pretrain.validation.compute_losses( model_200, batches, device = device )

# Compute wins from losses and batches.
timestamp_231 = pretrain.utils.get_timestamp_for_uid( 231 )
timestamp_200 = pretrain.utils.get_timestamp_for_uid( 200 )
for loss_231, loss_200 in list(zip( losses_231, losses_200 )):
    pretrain.validation.iswin( loss_231, loss_200, timestamp_231, timestamp_200 )
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
