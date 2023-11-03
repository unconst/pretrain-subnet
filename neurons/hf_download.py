from transformers import AutoModelForCausalLM, AutoTokenizer
import os

model_name = "mistralai/Mistral-7B-v0.1"
local_directory = '~/pretrain-subnet/neurons/pretraining_model/5EZGRiRtTuxipeSq1vgwZgkwu4kdj7Lh1U6Yu7ZerRMQkE19'

# Expand the user's home directory
local_directory = os.path.expanduser(local_directory)

# Ensure the directory exists
os.makedirs(local_directory, exist_ok=True)

# Download the model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Save the model and tokenizer to the specified directory
model.save_pretrained(local_directory)
tokenizer.save_pretrained(local_directory)
