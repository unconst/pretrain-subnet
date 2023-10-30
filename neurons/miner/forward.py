
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

import time
import typing
import pretrain
from rich import print
from rich.table import Table
import bittensor as bt

from helpers import compute_gradients_on_model

# === Blacklist ===
async def blacklist( self, synapse: pretrain.protocol.ComputeGradients ) -> typing.Tuple[bool, str]:
    # Locks requests to only allowing max_concurrent_forward_requests at a time.
    # After the blacklist the full synapse is pulled into memory so we want to limit
    # the number here.
    async with self.global_forward_lock:
        # Check if the hotkey is in the metagraph.
        if synapse.dendrite.hotkey not in self.metagraph.hotkeys:
                # Allow query through.
                return True, "Unrecognized hotkey"
        # Blacklist query.
        return False, "Hotkey recognized!"

# === Priority ===
async def priority( self, synapse: pretrain.protocol.ComputeGradients ) -> float:
    # Priority is stake based.
    caller_uid = self.metagraph.hotkeys.index( synapse.dendrite.hotkey )  
    prirority = float(self.metagraph.S[caller_uid]) 
    return prirority

# === Forward ===
async def compute_gradients( self, synapse: pretrain.protocol.ComputeGradients ) -> pretrain.protocol.ComputeGradients:
    """
    Compute the gradients for a given model based on passed batch, sequence length and pages.

    Args:
        synapse (pretrain.protocol.ComputeGradients): Object containing serialized model state 
                                                    and timeout value.

    Returns:
        pretrain.protocol.ComputeGradients: Object containing the serialized gradients.
    """
    try:
        start_call = time.time()
        forward_event = {}
        bt.logging.success(f'Received request for synapse: {synapse.axon.hotkey}')
        # Lock the model since concurrent accumulation to the model will poision the gradients we 
        # are computing. In practice we would shuttle multiple requests across multiple machines.
        # This lock is not necessary if we are only running a single cuda core.
        async with self.gpu_lock:
            # Lock the model since concurrent accumulation to the model will poision the gradients we
            start_process = time.time()
            bt.logging.debug( f'Aquired GPU space for query.' )

            # Move the model to the same device as the synapse
            local_model = pretrain.model.get_model()
            local_model.load_state_dict( synapse.deserialize() )
            local_model = local_model.to( self.config.device )
            
            # Compute gradients on the model.
            grads_dict, loss, n_tokens, n_examples, n_batches = compute_gradients_on_model(
                model = local_model,
                batch_size = synapse.batch_size,
                sequence_length = synapse.sequence_length,
                pages = synapse.pages
            )

            forward_event['loss'] = loss
            forward_event['n_tokens'] = n_tokens
            forward_event['n_examples'] = n_examples
            forward_event['n_batches'] = n_batches
            forward_event['pages'] = synapse.pages

        # Serialize accumulated gradients into the synapse object
        synapse.serialize( state_dict = grads_dict )
        bt.logging.debug( f'Serialized response gradients.' )

        # Log the state of the forward event
        forward_event['success'] = True
        forward_event['exception'] = synapse.pages
        forward_event['pages'] = synapse.pages
        forward_event['call_time'] = time.time() - start_call
        forward_event['process_time'] = time.time() - start_process

    except Exception as e:
        forward_event['success'] = False
        forward_event['exception'] = True
        bt.logging.error('Exception in forward: {}'.format(e))

    finally:
        # log_state
        forward_event['exception'] = False
        forward_event['success'] = False
        log_state( self, forward_event )
        return synapse


def log_state( self, forward_event: dict ):

    # Log the forward event to the console
    self.global_state['n_steps'] += 1
    self.global_state['n_successes'] += 1 if forward_event['success'] else 0
    self.global_state['n_failures'] += 0 if forward_event['success'] else 1
    self.global_state['n_exceptions'] += 1 if forward_event['exception'] else 0
    self.global_state['n_pages'] += len(forward_event['pages']) if 'pages' in forward_event else 0
    self.global_state['steps_per_second'] = 1 / (time.time() - self.global_state['last_query'])
    self.global_state['last_query'] = time.time()
    self.global_state['n_tokens'] += forward_event['n_tokens'] if 'n_tokens' in forward_event else 0
    self.global_state['n_examples'] += forward_event['n_examples'] if 'n_examples' in forward_event else 0
    self.global_state['n_batches'] += forward_event['n_batches'] if 'n_batches' in forward_event else 0
    self.global_state['loss'] = forward_event['loss'] if 'loss' in forward_event else 0.0

    # Create a log dictionary
    log = {
        'n_steps': self.global_state['n_steps'],
        'n_successes': self.global_state['n_successes'],
        'n_exceptions': self.global_state['n_exceptions'],
        'n_failures': self.global_state['n_failures'],
        'n_pages': self.global_state['n_pages'],
        'n_tokens': self.global_state['n_tokens'],
        'n_examples': self.global_state['n_examples'],
        'n_batches': self.global_state['n_batches'],
        'loss': self.global_state['loss'],
        'steps_per_second': self.global_state['steps_per_second'],
        'validation_time': forward_event['validation_time'] if 'validation_time' in forward_event else 0.0,
        'query_time': forward_event['query_time'] if 'query_time' in forward_event else 0.0,
        'forward_time': forward_event['forward_time'] if 'forward_time' in forward_event else 0.0,
        'step_time': forward_event['step_time'] if 'step_time' in forward_event else 0.0,
    }

    # Log using rich.
    table = Table()
    table.add_column("Metric", style="bold magenta")
    table.add_column("Value", style="bold green")
    table.add_row("Steps", str(log['n_steps']))
    table.add_row("Successes", str(log['n_successes']))
    table.add_row("Exceptions", str(log['n_exceptions']))
    table.add_row("Failures", str(log['n_failures']))
    table.add_row("Pages", str(log['n_pages']))
    table.add_row("Steps Per Second", f"{log['steps_per_second']:.2f}")
    table.add_row("Validation Time", f"{log['validation_time']:.2f}")
    table.add_row("Query Time", f"{log['query_time']:.2f}")
    table.add_row("Forward Time", f"{log['forward_time']:.2f}")
    table.add_row("Step Time", f"{log['step_time']:.2f}")
    table.add_row("Loss", f"{log['loss']:.2f}")
    table.add_row("Tokens", f"{log['n_tokens']:.2f}")
    table.add_row("Examples", f"{log['n_examples']:.2f}")
    table.add_row("Batches", f"{log['n_batches']:.2f}")
    print(table)

    # Log the forward event to wandb if configured
    if self.wandb:
        self.wandb.log( log )
