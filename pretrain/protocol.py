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

# --- Imports
import torch
import typing
import bittensor as bt

# --- Wire Protocol.
class ComputeGradients( bt.Synapse ):

    batch_size: int

    sequence_length: int

    pages: typing.List[int]
        
    state_dict: typing.Optional[ typing.Dict[ str, bt.Tensor ] ] = None

    def serialize( self, state_dict: typing.Dict[str, torch.Tensor ] ):
        self.state_dict = { key: bt.Tensor.serialize( value ) for key, value in state_dict.items() if value != None }

    def deserialize( self ) -> typing.Dict[str, torch.Tensor ]:
        return { key: value.deserialize() for key, value in self.state_dict.items() if value != None }
    

# --- GetState Protocol.
class GetState( bt.Synapse ):
        
    state_dict: typing.Optional[ typing.Dict[ str, bt.Tensor ] ] = None

    def serialize( self, state_dict: typing.Dict[str, torch.Tensor ] ):
        self.state_dict = { key: bt.Tensor.serialize( value ) for key, value in state_dict.items() if value != None }

    def deserialize( self ) -> typing.Dict[str, torch.Tensor ]:
        return { key: value.deserialize() for key, value in self.state_dict.items() if value != None }

