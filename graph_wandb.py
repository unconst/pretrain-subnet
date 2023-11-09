import json
import matplotlib.pyplot as plt

# Let's assume 'full_data' is a list of dictionaries loaded from "wandb_data.json",
# and each dictionary has the key 'original_format_json' with a JSON string as shown above.
with open("wandb_data.json", "r") as f:
    full_data = json.load(f)

# Initialize a dictionary to hold the average losses for each UID
uid_average_losses = {}

for entry in full_data:
    # Parse the JSON-formatted string to a Python dictionary
    step_info = json.loads(entry["original_format_json"])
    
    # Loop through each UID and extract the average loss
    for uid, uid_info in step_info["uid_data"].items():
        if uid not in uid_average_losses:
            uid_average_losses[uid] = []
        uid_average_losses[uid].append(uid_info["average_loss"])

# Now, let's plot the average losses for each UID
for uid, losses in uid_average_losses.items():
    plt.plot(losses, label=f'UID {uid}')

plt.xlabel('Step')
plt.ylabel('Average Loss')
plt.title('Average Loss over Steps for each UID')
plt.legend()
plt.show()