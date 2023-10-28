<div align="center">

# **Bittensor Training Subnet** <!-- omit in toc -->
[![Discord Chat](https://img.shields.io/discord/308323056592486420.svg)](https://discord.gg/bittensor)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

---

## Bittensor Incentivized Pretraining<!-- omit in toc -->

[Discord](https://discord.gg/bittensor) • [Network](https://taostats.io/) • [Research](https://bittensor.com/whitepaper)
</div>

---

## Running

Miners process produce gradients using their local machines. 
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

# Run first miner
python neurons/miner.py --wallet.name YOUR_MINER_COLD --wallet.hotkey YOUR_MINER_HOT_1 --device MINER_DEVICE_1 --axon.port AXON_PORT_1

# Run your second miner
python neurons/miner.py --wallet.name YOUR_MINER_COLD --wallet.hotkey YOUR_MINER_HOT_2 --device MINER_DEVICE_1 --axon.port AXON_PORT_2
```

Second run your validator/trainer on the same machine.
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
# Run the validator
python neurons/validator.py --wallet.name YOUR_VALIDATOR_COLD --wallet.hotkey YOUR_VALIDATOR_HOT --logging.debug --device YOUR_DEVICE
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
