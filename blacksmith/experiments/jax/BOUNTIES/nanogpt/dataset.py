import os
import requests
import numpy as np
# NO tiktoken. We want characters, not BPE.

input_file_path = os.path.join(os.path.dirname(__file__), 'data/input.txt')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    data = f.read()
print(f"length of dataset in characters: {len(data):,}")

# 2. Build the Vocabulary (Character-level)
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"all the unique characters: {''.join(chars)}")
print(f"vocab size: {vocab_size:,}") # Should be ~65

# Create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers

# 3. Tokenize
train_data = data[:int(len(data)*0.9)]
val_data = data[int(len(data)*0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)

print(f"Train has {len(train_ids):,} tokens.")
print(f"Val has {len(val_ids):,} tokens.")

# 4. Export to bin files
# We can use uint8 because vocab_size (65) < 255, but uint16 is safer standard
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'data/train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'data/val.bin'))

print("Data saved to data/train.bin and data/val.bin")

# SAVE THE META DATA FOR INFERENCE LATER
import pickle
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'data/meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)
print("Metadata saved to data/meta.pkl")