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

Bittensor subnet 9 rewards miners (engineers etc) for producing pretrained Foundation-Models on the Falcon Refined Web dataset. It acts like a continuous benchmark, whereby miners are paid out for attaining the best losses on randomly sampled pages of that dataset. The reward mechanism works as follows:

    1. Miner train and periodically host their model weights on a wandb account linked to their miner keys through the neurons/miner.py code. 
    2. Validators run a continuous eval on the hosted models, performing the validation system outlined in neurons/validator.py and setting weights to the chain based on the performance of each miner on the Falcon dataset.

#### Validation

Miners are evaluated based on the number of times their loss on a random batch from Falcon are lower than other miners. 
To perform well, miners must attain the lowest loss on the largest number of random batches sampled from the 900M page, 3T token dataset, Falcon Refined Wed.

All models are open and accessible via a wandb [project](https://wandb.ai/opentensor-dev/pretraining-subnet) and this repo contains tools for downloading them,
serving them on your own miner, as well as getting data from validators about which model perform best on which pages of the Falcon Dataset. Finding the best model at the earliest timestamp 
ensurs the most incentive.

```python
    epsilon = 0.01 # timestamp boost.
    while True:
        wins = {} # Count of wins per batch per miner

        # Run continous scoring until the epoch is over.
        while epoch_not_over( block )

            # Fetch random sample of batches to evaluate models on
            batches = get_random_sample_of_batches_from_falcon()
            
            # Fetch and or update models during this step.
            models = get_and_update_models_from_miners()

            # Compute losses for each batch and each minder
            losses = {}
            for model in models:
                for batch in batches:
                    loss = get_loss_for_model_on_batch( model, batch )
                    losses[ model ].append( loss )

            # Compute wins.
            wins = {}
            for model_a in models:
                for model_b in models:
                    for i in len( batches )
                        if losess[ model_a ][ i ] < losess[ model_b ][ i ]:
                            wins[ model_a ] += 1
                            
        # End epoch.
        # Weights are computed based on the ratio of wins a model attains during the epoch.
        weights = zeros()
        for model_i in models:
            weights = wins[ model_i ] / sum( wins.values() )

        # Set weights on the chain.
        set_weights( weight )
```

---

## Installing

Before mining make sure you have python3.8. Then install this repository.
```bash
git clone https://github.com/unconst/pretrain-subnet.git
cd pretrain-subnet
python -m pip install -e . 
```

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
following command.
```bash
# register your cold and associated hotkey to netuid 9
btcli s register --wallet.name ... --wallet.hotkey ... --netuid 0 
```

--- 
## Wandb

Miner and validators make heavy use of weights and biases in order to share model state and validation information. Both miners and validators must attain
a wandb account from [wandb](https://wandb.ai/home) along with their wandb api key which can be found by following the instructions [here](https://docs.wandb.ai/quickstart).

Models hosted by miners and corresponding validator information for runs can be found in this open wandb [project](https://wandb.ai/opentensor-dev/pretraining-subnet). You can get access to all valid, signed and recent miners runs from participants on the network as follows:

```python
>>> import pretrain
>>> import bittensor as bt
>>> meta = bt.subtensor(network = 'local' ).metagraph(9)
# Get all valid runs.
>>> miner_runs = pretrain.get_miner_runs( meta )
    {
        238: {
            'uid': 238, 
            'hotkey': '5CchHAvd95HtTaxfviiC36wt1HFXU73Xq9Aom7NDZJnAiG8v', 
            'emission': 0.02, 
            'run': <Run opentensor-dev/pretraining-subnet/63j2ps12 (finished)>, 
            'model_artifact': <File model.pth () 312.5MiB>, 
            'timestamp': 1699448922
        }, 
        239: {
            'uid': 239, 
            'hotkey': '5CSczy1dp4EpvLARaVbgvq8DST6oJgqmSTTQJZ8iXhJpKwdZ', 
            'emission': 0.01, 
            'run': <Run opentensor-dev/pretraining-subnet/qp0w790l (finished)>, 
            'model_artifact': <File model.pth () 312.5MiB>, 'timestamp': 1699448504
        } 
        ...
# Download model from run 1
>> model = pretrain.model.get_model() 
>> miner_runs['5CchHAvd95HtTaxfviiC36wt1HFXU73Xq9Aom7NDZJnAiG8v']['model_artifact'].download( replace=True, root=<path to model>)
>> model_weights = torch.load( <path to model> )
>> model.load_state_dict( model_weights )
```

Alternatively, you can download all validation data from wandb which can be used to evaluate how miners are performing on each individual page of the Falcon Refined Web dataset.
```python
>>> import pretrain
>>> import bittensor as bt
>>> meta = bt.subtensor(network = 'local' ).metagraph(9)
# Get all valid runs.
>>> vali_runs = pretrain.get_validator_runs( meta )
 {
        240: {
            'uid': 238, 
            'hotkey': '5CchHAvd95HtTaxfviiC36wt1HFXU73Xq9Aom7NDZJnAiG8v', 
            'stake': 123121, 
            'run': <Run opentensor-dev/pretraining-subnet/63j2ps12 (finished)>, 
        }, 
        ...
 }
 dataframe = vali_runs[240]['run'].history()
 ...
```

---

## Mining

The mining script can be run in multiple ways. In all cases, it uploads a model to wandb which will be evaluated by validators. 

By default, the miner trains from scratch and posts the model periodically as its loss decreases on Falcon.
```bash
python neurons/miner.py --wallet.name ... --wallet.hotkey ... --num_epochs 10 --pages_per_epoch 5
```

Alternatively, you can scrape a model from an already miner on the network by passing its run id. This starts the training process from the checkpoint of another 
miner.
```bash
python neurons/miner.py --wallet.name ... --wallet.hotkey ... --num_epochs 10 --pages_per_epoch 5 --load_run_id ...
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
