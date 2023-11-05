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
            models = get_and_update_models_from_miners()

            # Compute losses for each batch on subset and count wins per miner
            for batch in batches:

                # Find miner with lowest loss on the batch.
                for miner_uid, model in enumerate( models ):
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
btcli s recycle_register --wallet.name ... --wallet.hotkey ... --netuid 0 
```

---

## Mining

Miners must attain a wandb account from [wandb](https://wandb.ai/home) and attain their wandb api key.
Follow the instructions [here](https://docs.wandb.ai/quickstart) to get access to your API key for logging in.

```bash
wandb init
```

Once your wandb is installed. Train your model. Note, the training script is simply a mock training script, we recommend you ammend the training script at a later date.
You can directly pass a ```--model_path ``` flag to have the training script write your model under a newly specified path. By default the trained model is written to 
```~/.bittensor/miners/<your cold>/<your hot>/netuid9/miner/model.pth```. Note if you pass your model path it must end in ```model.pth```.

```bash
# Run your miner with specified wallet keys.
python neurons/train.py --wallet.name ... --wallet.hotkey ... --num_epochs 1 --pages_per_epoch 1
... training ....
2023-11-04 13:53:04.134 |     SUCCESS      | Saving model to /home/setup/.bittensor/miners/my_coldkey/my_hotkey/netuid9/miner/model.pth
```

To host your model on the miner run the following script. By default the miner will host the model stored under ```~/.bittensor/miners/<your cold>/<your hot>/netuid9/miner/model.pth```
however pass ```--model_path ``` to set an alterative model path. The miner will use your wandb run to store the model artifact which becomes available for validators to validate.
You can follow the wandb run link to see model artifact.

```bash
python neurons/miner.py --wallet.name ... --wallet.hotkey ...
2023-11-04 13:55:44.141 |     SUCCESS      | Started wandb run
2023-11-04 13:55:46.798 |     SUCCESS      | Loaded model from: /home/setup/.bittensor/miners/Rawls/M1/netuid9/miner/model.pth
2023-11-04 13:56:27.397 |     SUCCESS      | Waiting for updated on /home/setup/.bittensor/miners/my_coldkey/my_hotkey/netuid9/miner/model.pth
...
```

---

## Validating

First follow the instructions to install and create your wallet before proceeding. Validators should also use wandb to help us monitor the training runs.
Follow the instructions [here](https://docs.wandb.ai/quickstart) to get access to your API key for logging in.

```bash
wandb init
```

Next run your validator as follows, passing the name of your validator codld and hotkeys to the python script. Note validation required you have a working GPU.
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
