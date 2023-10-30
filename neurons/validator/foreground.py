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
import torch
import random
import asyncio
import helpers
import pretrain
import bittensor as bt
from rich import print
from rich.table import Table

# === Training loop ===
async def foreground_loop(self: object):
    """
        Initiates a training loop that continuously performs forward passes,
        logs the events, and updates the global state asynchronously.

        This method starts a validator training loop that concurrently handles
        forward calls, ensuring controlled concurrency using a lock. For each iteration,
        a task is created for a forward pass, the global state is updated, and the event
        is logged to wandb if configured. A short wait is introduced before proceeding
        to the next iteration to manage the rate of task creation.
    """
    # Log the initiation of the training loop
    bt.logging.debug('Starting validator training loop.')
    
    while True:
        # Control concurrency of forward calls using a lock
        async with self.forward_lock:
            
            # Create a task for a forward pass
            asyncio.create_task(forward(self))
                        
            # Introduce a short wait before the next iteration
            await asyncio.sleep(1)

# === Forward Function ===
async def forward(self: object) -> dict:
    """
        Perform a forward pass through a randomly selected miner axon,
        compute gradients both locally and through a request to the miner,
        compare them to score the miner, and then apply the received gradients
        to the local model.

    Returns:
        dict: A dictionary containing details of the forward event, including
                any exceptions that may have occurred.

    """
    # Initialize forward event dictionary and record start time
    forward_event = {}
    start_forward = time.time()
    bt.logging.debug(f'Starting forward call')

    try:
        # Get uid to query
        uid = next_uid_to_query(self)
        forward_event['uid'] = uid

        # Query the miner
        begin_model_state = self.model.state_dict()
        response = await query_uid(self, uid, begin_model_state, forward_event)

        # Check for successful response
        if response.is_success:
            # Deserialize gradients and validate against local computation
            grads_dict = response.deserialize()
            mse_score = validate_gradients(self, uid, grads_dict, begin_model_state, forward_event)
            update_score(self, uid, mse_score, forward_event)

            # Apply received gradients to local model
            await step_with_gradients(self, grads_dict, forward_event)

        else:
            # Update score negatively for failure
            update_score(self, uid, -10, forward_event)

    # Log and record exceptions
    except Exception as e:
        bt.logging.error(f'Caught exception during forward with error:: {e}')
        forward_event['exception'] = True

    # Log success and return forward event details
    finally:
        bt.logging.debug(f'Finished forward to uid: {uid} call with success.')
        forward_event['forward_time'] = time.time() - start_forward
        forward_event['exception'] = False
        log_state( self, forward_event )
        return forward_event


# === Returns next uid to query ===
def next_uid_to_query( self: object ) -> int:
    """
        Get the next miner UID to query from the available set of UIDs where
        availability is determined based on whether the corresponding axon is serving.
        In case there are no available UIDs, an exception is raised.        
        Returns:
            int: The next UID to query.
        Raises:
            Exception: If there are no available UIDs.
    """
    # Get a random miner axon.
    available_uids = [uid.item() for uid in self.metagraph.uids if self.metagraph.axons[uid].is_serving]
    if len(available_uids) == 0:
        raise Exception('No available uids.')
    uid = random.choice(available_uids)
    return uid

# === Update UID score ===
def update_score(self: object, uid: int, score: float, forward_event: dict):
    """
        Updates the score of a given UID using a moving average formula.
        The weighting factor is determined by the alpha value from the         
        Args:
            uid (int): The UID whose score is to be updated.
            score (float): The new score to be considered for the update.                                    
    """
    # Update the score using a moving average formula
    self.scores[ uid ] = self.config.alpha * self.scores[uid] + ( 1 - self.config.alpha ) * score if uid in self.scores else self.config.alpha * score
    # Update the forward_event dictionary with the new and current scores
    bt.logging.debug( f'Updated score for uid: {uid} with score: {score} to {self.scores[ uid ]}' )
    forward_event['score'] = score
    forward_event['current_score'] = self.scores[ uid ]

# === Query uid ===
async def query_uid(self: object, uid: int, model_state: dict, forward_event: dict ):
    """
        Asynchronously query a specified UID with a given model state and log the event.
        A random page is selected as argument for the query.
        Args:
            uid (int): The UID to be queried.
            model_state (dict): The state dictionary of the model to be used in the query.
            forward_event (dict): A dictionary to which the event details will be added.
        Returns:
            response: The response object returned from the query.
    """
    # === Lock total concurrent forward calls per uid ===
    async with self.per_uid_locks[uid]:

        # Log the initiation of the query
        bt.logging.debug(f'Making forward query to uid: {uid}')
        
        # Build the query
        pages = [random.randint(1, 968000015)]
        synapse = pretrain.protocol.ComputeGradients(
            batch_size=self.config.batch_size,
            sequence_length=self.config.sequence_length,
            pages=pages
        )
        synapse.serialize(state_dict=model_state)
        
        # Issue the query and record the start time
        start_query = time.time()
        dendrite = bt.dendrite(wallet=self.wallet)
        response = await dendrite.forward(self.metagraph.axons[uid], synapse, timeout=360, deserialize=False)
        await dendrite.close_session()
        
        # Log and record the result of the query
        forward_event['pages'] = pages
        forward_event['query_time'] = time.time() - start_query
        
        if not response.is_success:
            # Log and record a failed query
            bt.logging.error(f'Forward query was a failure with error: {response.axon.status_code}:{response.axon.status_message}')
            forward_event['success'] = False
        else:
            # Log and record a successful query
            forward_event['success'] = True
            bt.logging.debug(f'Forward query was a success with process time: {response.axon.process_time }s')
        
        # Return the response object
        return response

async def validate_gradients(self: object, grads_dict: dict, model_state: dict, forward_event: dict):
    """
    Asynchronously validate the gradients received from a query by recomputing them locally.
    It computes the Mean Squared Error (MSE) between the received gradients and the locally computed gradients. 
    Args:
        grads_dict (dict): The gradients dictionary received from a query.
        model_state (dict): The state dictionary of the model to be used for local gradient computation.
        forward_event (dict): A dictionary to which the event details will be added.
    Returns:
        float: The negative MSE score between the received gradients and the locally computed gradients.
    """
    # Deserialize the gradients from the response
    uid = forward_event['uid']
    bt.logging.debug(f'Validating gradients from uid: {uid}')
    start_validate = time.time()
    
    # Recompute the gradients using the local model state
    async with self.gpu_lock:
        # Using GPU, create local gradient replicas.
        # This deallocates all the GPU memory used in the following code block.
        with torch.cuda.device(self.config.device):
            # Put eval model on the GPU
            eval_model = pretrain.model.get_model()
            eval_model.load_state_dict(model_state)
            eval_model.to(self.config.device)

            # Compute local gradients on the model
            local_grads, loss, n_tokens, n_examples, n_batches = self.helpers.compute_gradients_on_model(
                model=eval_model,
                batch_size=self.config.batch_size,
                sequence_length=self.config.sequence_length,
                pages=forward_event['pages']
            )

            # Delete eval model as a safety check
            del eval_model
    
    # Compute the Mean Squared Error between the received and locally computed gradients
    mse_score = -helpers.mse_gradients(local_grads, grads_dict)
    
    # Log and record the result of the gradient validation
    forward_event['validation_loss'] = loss
    forward_event['validation_time'] = time.time() - start_validate
    
    # Return the negative MSE score
    return mse_score

async def step_with_gradients(self: object, grads_dict: dict, forward_event: dict ):
    """
        Asynchronously apply the gradients received from a query to the local model.
        his method takes the gradients from a specified uid contained in the
        grads_dict, and applies them to the local model using the configured optimizer.
    
        Args:
            grads_dict (dict): The gradients dictionary received from a query.
            forward_event (dict): A dictionary to which the event details will be added.
    """
    # Log the initiation of applying gradients
    uid = forward_event['uid']
    bt.logging.debug(f'Applying gradients from: {uid}')
    start_step = time.time()
    
    async with self.model_lock:
        # Accumulate grads on local model
        self.model.to('cpu')
        for name_j, param_j in self.model.named_parameters():
            if name_j in grads_dict and grads_dict[name_j] is not None:
                param_j.grad = param_j.grad + grads_dict[name_j].to('cpu') if param_j.grad is not None else grads_dict[name_j].to('cpu')

        # Take a step using the optimizer to update model parameters
        self.optimizer.step()
        bt.logging.debug(f'Applied step.')

        # Zero out the gradients to free up memory and avoid accidental accumulation in the next iteration
        self.model.zero_grad()
        forward_event['step_time'] = time.time() - start_step


def log_state( self, forward_event: dict ):

    # Log the forward event to the console
    self.global_state['n_steps'] += 1
    self.global_state['n_successes'] += 1 if forward_event['success'] else 0
    self.global_state['n_failures'] += 0 if forward_event['success'] else 1
    self.global_state['n_exceptions'] += 1 if forward_event['exception'] else 0
    self.global_state['n_pages'] += len(forward_event['pages'])
    self.global_state['steps_per_second'] = 1 / (time.time() - self.global_state['last_query'])
    self.global_state['last_query'] = time.time()

    # Create a log dictionary
    log = {
        'n_steps': self.global_state['n_steps'],
        'n_successes': self.global_state['n_successes'],
        'n_exceptions': self.global_state['n_exceptions'],
        'n_failures': self.global_state['n_failures'],
        'n_pages': self.global_state['n_pages'],
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
    print(table)

    # Log the forward event to wandb if configured
    if self.wandb:
        self.wandb.log( log )




