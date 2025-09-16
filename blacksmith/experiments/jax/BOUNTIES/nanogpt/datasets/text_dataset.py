# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
import pickle
import requests
import jax
import jax.numpy as jnp
from typing import Tuple, Optional, List
import numpy as np
from tqdm import tqdm


class SimpleTokenizer:
    """Simple character-level tokenizer for text data."""
    
    def __init__(self, vocab_size: int = 50304):
        self.vocab_size = vocab_size
        self.chars = None
        self.stoi = None
        self.itos = None
        self._build_vocab()
    
    def _build_vocab(self):
        """Build vocabulary from common characters."""
        # Common characters in English text
        chars = list(""" !"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~""")
        
        # Add some special tokens
        chars.extend(['\n', '\t', '\r'])
        
        # Pad with additional characters if needed
        while len(chars) < self.vocab_size:
            chars.append(f'<unk_{len(chars)}>')
        
        # Truncate if too many
        chars = chars[:self.vocab_size]
        
        self.chars = chars
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        return [self.stoi.get(ch, self.stoi.get('<unk_0>', 0)) for ch in text]
    
    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs to text."""
        return ''.join([self.itos.get(token, '<unk>') for token in tokens])


class TextDataset:
    """Dataset class for text data with tokenization."""
    
    def __init__(self, data_dir: str = "data", block_size: int = 1024, vocab_size: int = 50304):
        self.data_dir = data_dir
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.tokenizer = SimpleTokenizer(vocab_size)
        self.data = None
        self.train_data = None
        self.val_data = None
        
    def prepare_data(self, dataset_name: str = "openwebtext"):
        """Prepare and tokenize the dataset."""
        if dataset_name == "openwebtext":
            self._prepare_openwebtext()
        elif dataset_name == "shakespeare":
            self._prepare_shakespeare()
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    def _prepare_shakespeare(self):
        """Prepare Shakespeare dataset for testing."""
        # Download Shakespeare text
        shakespeare_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        
        os.makedirs(self.data_dir, exist_ok=True)
        shakespeare_path = os.path.join(self.data_dir, "shakespeare.txt")
        
        if not os.path.exists(shakespeare_path):
            print("Downloading Shakespeare dataset...")
            response = requests.get(shakespeare_url)
            with open(shakespeare_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
        
        # Read and tokenize
        with open(shakespeare_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        # Split into train/val
        n = len(tokens)
        train_data = tokens[:int(n * 0.9)]
        val_data = tokens[int(n * 0.9):]
        
        self.train_data = jnp.array(train_data, dtype=jnp.int32)
        self.val_data = jnp.array(val_data, dtype=jnp.int32)
        
        print(f"Shakespeare dataset prepared: {len(train_data)} train tokens, {len(val_data)} val tokens")
    
    def _prepare_openwebtext(self):
        """Prepare OpenWebText dataset (simplified version)."""
        # For this implementation, we'll create a synthetic dataset
        # In a real implementation, you would download and process OpenWebText
        
        print("Creating synthetic OpenWebText-like dataset...")
        
        # Generate synthetic text data
        np.random.seed(42)
        vocab_size = self.vocab_size
        
        # Create synthetic tokens that mimic real text distribution
        n_tokens = 10_000_000  # 10M tokens
        tokens = np.random.randint(0, vocab_size, size=n_tokens, dtype=np.int32)
        
        # Add some structure to make it more realistic
        # Add some repeated patterns
        for i in range(0, n_tokens - 100, 1000):
            pattern = np.random.randint(0, vocab_size, size=10)
            tokens[i:i+10] = pattern
        
        # Split into train/val
        n = len(tokens)
        train_data = tokens[:int(n * 0.9)]
        val_data = tokens[int(n * 0.9):]
        
        self.train_data = jnp.array(train_data, dtype=jnp.int32)
        self.val_data = jnp.array(val_data, dtype=jnp.int32)
        
        print(f"Synthetic dataset prepared: {len(train_data)} train tokens, {len(val_data)} val tokens")
    
    def get_batch(self, split: str, batch_size: int, device: Optional[jax.Device] = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get a batch of data for training/validation."""
        data = self.train_data if split == 'train' else self.val_data
        
        # Generate random starting indices
        ix = jax.random.randint(
            jax.random.PRNGKey(0), 
            (batch_size,), 
            0, 
            len(data) - self.block_size
        )
        
        # Create batches
        x = jnp.stack([data[i:i+self.block_size] for i in ix])
        y = jnp.stack([data[i+1:i+self.block_size+1] for i in ix])
        
        # Move to device if specified
        if device is not None:
            x = jax.device_put(x, device)
            y = jax.device_put(y, device)
        
        return x, y
    
    def get_batch_cpu(self, split: str, batch_size: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get a batch of data on CPU (for fallback scenarios)."""
        data = self.train_data if split == 'train' else self.val_data
        
        # Generate random starting indices on CPU
        with jax.default_device(jax.devices("cpu")[0]):
            key = jax.random.PRNGKey(0)
            ix = jax.random.randint(key, (batch_size,), 0, len(data) - self.block_size)
            
            # Create batches
            x = jnp.stack([data[i:i+self.block_size] for i in ix])
            y = jnp.stack([data[i+1:i+self.block_size+1] for i in ix])
        
        return x, y


def load_text_dataset(config) -> TextDataset:
    """Load text dataset from configuration."""
    dataset = TextDataset(
        data_dir=config.data.data_dir,
        block_size=config.data.block_size,
        vocab_size=config.model.vocab_size
    )
    
    dataset.prepare_data(config.data.dataset)
    return dataset


def create_dataloader(dataset: TextDataset, config, device: Optional[jax.Device] = None):
    """Create a data loader for the dataset."""
    def get_batch_fn(split: str):
        def _get_batch():
            if device is not None and device.platform == 'tt':
                # Use CPU fallback for TT device data loading
                return dataset.get_batch_cpu(split, config.data.batch_size)
            else:
                return dataset.get_batch(split, config.data.batch_size, device)
        return _get_batch
    
    return {
        'train': get_batch_fn('train'),
        'val': get_batch_fn('val')
    }
