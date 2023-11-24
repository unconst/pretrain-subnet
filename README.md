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

Bittensor subnet 9 rewards miners (engineers etc) for producing pretrained Foundation-Models on the Falcon Refined Web dataset. It acts like a continuous benchmark, whereby miners are paid out for attaining the best losses on randomly sampled pages of that dataset and co-temporally producing best deltas over the 'head' model. The reward mechanism works as follows:

    1. Miner train and periodically host their model weights on a wandb account linked to their miner keys through the neurons/miner.py code.
    1. Miner train and periodically host a 'delta', a weights pertubation, on the same wandb account. The delta is assumed to apply to the 'head' model, which is the miner with the highest network incentive score.
    2. Validators run a continuous eval on the hosted models, performing the validation system outlined in neurons/validator.py and setting weights to the chain based on the performance of each miner on the Falcon dataset.

#### Validation

Miners are evaluated based on the number of times their loss on a random batch from Falcon are lower than other miners and how their delta over the head model performs.
To perform well, miners must attain the lowest loss on the largest number of random batches sampled from the 900M page, 3T token dataset, Falcon Refined Wed for both their own models, and their delta.

All models + deltas are open and accessible via a wandb [project](https://wandb.ai/opentensor-dev/pretraining-subnet) and this repo contains tools for downloading them,
serving them on your own miner, as well as getting data from validators about which model perform best on which pages of the Falcon Dataset. Finding the best model and delta at the earliest timestamp 
ensurs the most incentive.

```python
    while True:
        wins = {} # Count of wins per batch per miner

        # Run continous scoring until the epoch is over.
        while epoch_not_over( block )

            # Fetch random sample of batches to evaluate models on
            batches = get_random_sample_of_batches_from_falcon()
            
            # Fetch and or update models during this step.
            models = get_and_update_models_from_miners()

            # Fetch and or update deltas during this step.
            deltas = get_and_deltas_from_miners()

            # Fetch the model with the highest incentive on the network.
            head_model = get_highest_incentive_model()

            # Compute losses for each batch and each model
            model_losses = {}
            for model in models:
                for batch in batches:
                    loss = get_loss_for_model_on_batch( model, batch )
                    model_losses[ model ].append( loss )
        
            # Compute losses for each batch and each model
            delta_losses = {}
            for delta in deltas:
                for batch in batches:
                    perturbed_head = apply_delta( head_model,  delta )
                    loss = get_loss_for_model_on_batch( perturbed_head, batch )
                    delta_losses[ model ].append( loss )

            # Compute wins for models.
            model_wins = {}
            for model_a in models:
                for model_b in models:
                    for i in len( batches )
                        # Determine if better model loss with timestamp boosting.
                        if iswin( model_losses[ model_a ][ i ], model_losses[ model_b ][ i ], timestamp_a, timestamp_b ):
                            model_wins[ model_a ] += 1

            # Compute wins for deltas.
            delta_wins = {}
            for delta_a in deltas:
                for delta_b in deltas:
                    for i in len( batches )
                        # Determine if better delta loss with timestamp boosting.
                        if iswin( delta_losses[ model_a ][ i ], delta_losses[ model_b ][ i ], timestamp_a, timestamp_b ):
                            delta_wins[ model_a ] += 1
                            
        # End epoch.
        # Weights are computed based on the ratio of wins a model attains during the epoch.
        weights = zeros()
        for model_i in models:
            weights += model_wins[ model_i ] / sum( model_wins.values() )
        for delta_i in deltas:
            weights += delta_wins[ delta_i ] / sum( delta_wins.values() )

        # Set weights on the chain.
        set_weights( weight )
```

The behaviour of the `iswin( loss_a, loss_b, timestamp_a, timestamp_b)` function is integral to the way in which this incentive function works since it skews the win function 
to reward models and deltas which have been hosted earlier on wandb. Specifically, a newer model or delta is only better than another if it is more than `epsilon` percent lower accoring to
the following function.
```python

def iswin( loss_a, loss_b, timestamp_a, timestamp_b, epsilon ):
    loss_a = (1 - epsilon) * loss_a if timestamp_a < timestamp_b else loss_a
    loss_b = (1 - epsilon) * loss_b if timestamp_b < timestamp_a else loss_b
    if loss_a < loss_b: return True
    else: return False
```

It is important to note that this effects the game theoretics of the incentive landscape since miners should only update their delta or model (thus updating their timestamp to a newer date) if they
have a achieved an `epsilon` better loss on average on the Falcon Refined Web dataset. This undermines the obvious optimal strategy for miners to simple copy the publicly available models and deltas
of other miners. They **can** and should copy other miners, but they will always obtain fewer wins compared to them until they also decrease their loss by `epsilon`. The effect is to drive miners to 
continually produce better models.

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
following command. If you dont have any TAO message const [t,t] on discord for a faucet to try things out, please dont scam me.
```bash
# register your cold and associated hotkey to netuid 9
btcli s register --wallet.name ... --wallet.hotkey ... --netuid 0 
```

--- 
## Tools

The Pretraining package comes with some helper functions to enable better miner performance.

Getting models and deltas from wandb based on a miner uid:
```python
import pretrain
device = 'cuda'

# Downloads and retrieves the torch model hosted on wandb for a miner within the set of available 0->255
model = pretrain.utils.get_and_update_model_for_uid( 231, device = device )

# Downloads the retrieves the torch delta hosted on wandb for a miner within the available 0->255
delta = pretrain.utils.get_and_update_delta_for_uid( 102, device = device )

# Applies the delta to the model in the same manner performed by validators.
perturbed_model = pretrain.utils.apply_delta( model, delta, device = device )

# Attains batches from the Falcon Dataset based on pages 
pages = [random.randint(1, pretrain.dataset.SubsetFalconLoader.max_pages) for _ in range(self.config.pages_per_eval)]
batches = {
    page: list(pretrain.dataset.SubsetFalconLoader(
        batch_size = pretrain.batch_size,
        sequence_length = pretrain.sequence_length,
        pages = [page]
    )) for page in pages
}

# Evaluate the models on these batches.
losses = pretrain.validation.compute_losses( perturbed_model, batches, device = device )
```


--- 
## Wandb

Miner and validators make heavy use of weights and biases in order to share model state and validation information. Both miners and validators must attain
a wandb account from [wandb](https://wandb.ai/home) along with their wandb api key which can be found by following the instructions [here](https://docs.wandb.ai/quickstart).

---

## Mining

The mining script can be run in multiple ways. In all cases, it uploads a model to wandb which will be evaluated by validators. 

By default, the miner trains from scratch and posts the model periodically as its loss decreases on Falcon.
```bash
python neurons/miner.py --wallet.name ... --wallet.hotkey ... --num_epochs 10 --pages_per_epoch 5
```

Alternatively, you can scrape a model from an already running miner on the network by passing its run id. This starts the training process from the checkpoint of another 
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
