from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import subprocess

model_name = "facebook/opt-125m"
local_directory = '~/pretrain-subnet/neurons/pretraining_model/5EZGRiRtTuxipeSq1vgwZgkwu4kdj7Lh1U6Yu7ZerRMQkE19'

# Expand the user's home directory
local_directory = os.path.expanduser(local_directory)

# Ensure the directory exists
os.makedirs(local_directory, exist_ok=True)

# Define the path to the directory whose contents you want to remove
dir_to_remove = os.path.join(local_directory, "pretraining_model/5EZGRiRtTuxipeSq1vgwZgkwu4kdj7Lh1U6Yu7ZerRMQkE19/*")

# Use subprocess to run the rm command
try:
    subprocess.run(['rm', '-rf', dir_to_remove], check=True)
except subprocess.CalledProcessError as e:
    print(f"An error occurred while trying to remove files: {e}")

# Download the model and tokenizer with trust_remote_code set to True
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# Save the model and tokenizer to the specified directory
model.save_pretrained(local_directory)
tokenizer.save_pretrained(local_directory)
