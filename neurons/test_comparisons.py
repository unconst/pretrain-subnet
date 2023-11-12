# Data setup for the example scenario
uids = [1, 2, 3]
page_j = 1  # Assuming a single page for simplicity

# Losses for each UID for 10 batches
losses_per_page_per_uid = {
    1: {page_j: [0.9, 1.1, 1.0, 1.2, 0.8, 1.0, 1.1, 1.3, 1.0, 0.9]},
    2: {page_j: [1.0, 1.0, 1.2, 1.1, 0.9, 1.1, 1.2, 1.0, 1.1, 1.0]},
    3: {page_j: [1.1, 0.9, 1.1, 1.0, 1.0, 1.2, 0.9, 1.1, 1.2, 1.1]}
}

# Timestamps for each UID
metadata = {
    1: {'timestamp': 100},
    2: {'timestamp': 200},
    3: {'timestamp': 150}
}

# Function to determine winning loss with timestamps
def is_winning_loss_with_timestamps(this_uid, page_j, batch_k):
    this_loss = losses_per_page_per_uid[this_uid][page_j][batch_k]
    this_timestamp = metadata[this_uid]['timestamp']
    for other_uid in uids:
        if other_uid != this_uid:
            other_loss = losses_per_page_per_uid[other_uid][page_j][batch_k]
            other_timestamp = metadata[other_uid]['timestamp']
            if this_timestamp > other_timestamp:
                other_loss *= (1 - 0.03)
            elif this_timestamp < other_timestamp:
                this_loss *= (1 - 0.03)
            if this_loss > other_loss:
                return False
    return True

# Executing the logic for each batch
win_counts = {uid: 0 for uid in uids}
for batch_k in range(10):
    for uid in uids:
        if is_winning_loss_with_timestamps(uid, page_j, batch_k):
            win_counts[uid] += 1

print(win_counts)
