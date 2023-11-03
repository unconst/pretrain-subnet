from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "EleutherAI/gpt-neo-125m"
local_directory = '~/pretrain-subnet/neurons/pretraining_model/5EZGRiRtTuxipeSq1vgwZgkwu4kdj7Lh1U6Yu7ZerRMQkE19'

# Expand the user's home directory
local_directory = os.path.expanduser(local_directory)

# Ensure the directory exists
os.makedirs(local_directory, exist_ok=True)

# Download the model and tokenizer with trust_remote_code set to True
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, cache_dir=local_directory)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, cache_dir=local_directory)

# Save the model and tokenizer to the specified directory
model.save_pretrained(local_directory)
tokenizer.save_pretrained(local_directory)
