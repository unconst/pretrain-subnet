
import wandb
import bittensor as bt
from tqdm import tqdm
from datetime import datetime

api = wandb.Api( timeout = 100 )

def get_valid_runs( metagraph ):
    runs = api.runs("opentensor-dev/openpretraining")
    pbar = tqdm( runs , desc="Getting runs:", leave=False )
    valid_runs = {}
    model_timestamps = {}
    for run in pbar:
        pbar.set_description(f"Checking: {run.id}")

        # Find hotkey of continue
        if 'hotkey' not in run.config: continue
        hotkey = run.config['hotkey']

        # Filter models not registered
        if hotkey not in metagraph.hotkeys: continue
        uid = metagraph.hotkeys.index( hotkey )

        # Find signature or continue
        if 'signature' not in run.config: continue
        signature = run.config['signature']

        # Check signature
        keypair = bt.Keypair( ss58_address = hotkey )
        is_legit = keypair.verify( run.id, bytes.fromhex( signature ) )
        if not is_legit: continue

        # Check for model
        try:
            model_artifact = run.file('model.pth')
        except: continue

        # Check if it is the latest model
        model_timestamp = int(datetime.strptime(model_artifact.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
        if hotkey in model_timestamps and model_timestamps[hotkey] > model_timestamp:
            continue

        # Set run as valid with and latest.
        valid_runs[hotkey] = {'run': run, 'model_artifact': model_artifact, 'timestamp': model_timestamp, 'uid': uid, 'emission': metagraph.E[uid].item() }  

    return valid_runs

sub = bt.subtensor( network = 'local' )
meta = sub.metagraph( 9 )
print( get_valid_runs( meta ) )



# def get_valid_runs( metagraph ):
#     valid_runs_per_hotkey = {}
#     run_timestamp_per_hotkey = {}
#     runs = api.runs("opentensor-dev/openpretraining")

#     pbar = tqdm( runs , desc="Updating model:", leave=False )
#     for run in runs:

#         # Get run tags and cehck format.
#         tags = run.tags
#         run_id = run.id
#         if len( tags ) != 4: continue 
        
#         # Check signature.
#         run_uid = tags[0]
#         run_hotkey = tags[1]
#         run_signature = tags[2] + tags[3]
#         run_signature  = bytes.fromhex( run_signature )
#         keypair = bt.Keypair( ss58_address = run_hotkey )
#         is_legit = keypair.verify( run_id, run_signature )
#         if not is_legit: continue

#         # Check if hotkey is relevant
#         if run_hotkey not in metagraph.hotkeys:
#             continue

#         model_file = run.file(ARTIFACT_NAME)

#         # Check if latest run.
#         model_timestamp = int(datetime.strptime(model_file.updatedAt, '%Y-%m-%dT%H:%M:%S').timestamp())
#         if run_hotkey in run_timestamp_per_hotkey:
#             if run.created_at < run_timestamp_per_hotkey[ run_hotkey ]:
#                 continue

#          # === Check if the timestamp file exists and if the timestamp matches ===
#         if os.path.exists(timestamp_file):
#             with open(timestamp_file, 'r') as f:
#                 existing_timestamp = json.load(f)
#             if existing_timestamp == model_timestamp:
#                 bt.logging.debug('Miner model artifact is up to date.')
#                 return

#         # Sink run to map.
#         valid_runs_per_hotkey[run_hotkey] = run
#         run_timestamp_per_hotkey[run_hotkey] = run.created_at

#         uid = metagraph.hotkeys.index( run_hotkey )

#         # === Set the paths ===
#         model_dir = f'{config.full_path}/models/{run_hotkey}/'

#         # === Check if the timestamp file exists and if the timestamp matches ===
#         timestamp_file = f'{model_dir}timestamp.json'
#         if os.path.exists(timestamp_file):
#             with open(timestamp_file, 'r') as f:
#                 existing_timestamp = json.load(f)
#             if existing_timestamp == model_timestamp:
#                 bt.logging.debug('Miner model artifact is up to date.')
#                 return

#         # === If the timestamp does not match or the file does not exist, update the timestamp ===
#         os.makedirs(model_dir, exist_ok=True)  # Ensure the directory exists
#         with open(timestamp_file, 'w') as f:
#             json.dump(model_timestamp, f)

#         model_paths[uid] = model_dir + ARTIFACT_NAME
#         model_timestamps[uid] = model_timestamp

#         # === Load the model from the file ===
#         bt.logging.debug(f"Updating model for: {uid}")
#         model_file.download(replace=True, root=model_dir)