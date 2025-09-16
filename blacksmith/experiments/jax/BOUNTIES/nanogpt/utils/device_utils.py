# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import jax
import jax.numpy as jnp
from typing import Any, Callable, Optional, Tuple
import functools
import logging


class DeviceManager:
    """Manages device operations and fallback mechanisms."""
    
    def __init__(self, primary_device: str = "tt", fallback_device: str = "cpu"):
        self.primary_device = primary_device
        self.fallback_device = fallback_device
        self.cpu_device = jax.devices("cpu")[0]
        self.tt_device = None
        
        # Try to get TT device
        try:
            tt_devices = jax.devices("tt")
            if tt_devices:
                self.tt_device = tt_devices[0]
        except Exception as e:
            logging.warning(f"TT device not available: {e}")
            self.tt_device = None
        
        # Set current device
        if primary_device == "tt" and self.tt_device is not None:
            self.current_device = self.tt_device
        else:
            self.current_device = self.cpu_device
            self.primary_device = "cpu"
    
    def get_device(self, device_type: str = None) -> jax.Device:
        """Get device of specified type."""
        if device_type is None:
            return self.current_device
        elif device_type == "cpu":
            return self.cpu_device
        elif device_type == "tt" and self.tt_device is not None:
            return self.tt_device
        else:
            return self.cpu_device
    
    def device_put(self, data: Any, device_type: str = None) -> Any:
        """Put data on specified device."""
        device = self.get_device(device_type)
        return jax.device_put(data, device)
    
    def with_device(self, device_type: str):
        """Context manager for setting default device."""
        return jax.default_device(self.get_device(device_type))
    
    def is_tt_available(self) -> bool:
        """Check if TT device is available."""
        return self.tt_device is not None


def safe_jit(fn: Callable, device_manager: DeviceManager, fallback_to_cpu: bool = True):
    """Safely JIT compile a function with fallback to CPU if TT fails."""
    
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if device_manager.primary_device == "tt" and device_manager.is_tt_available():
            try:
                # Try to run on TT device
                with device_manager.with_device("tt"):
                    return fn(*args, **kwargs)
            except Exception as e:
                logging.warning(f"TT device operation failed: {e}")
                if fallback_to_cpu:
                    logging.info("Falling back to CPU")
                    with device_manager.with_device("cpu"):
                        return fn(*args, **kwargs)
                else:
                    raise
        else:
            # Run on CPU
            with device_manager.with_device("cpu"):
                return fn(*args, **kwargs)
    
    return wrapper


def safe_compile(fn: Callable, device_manager: DeviceManager, fallback_to_cpu: bool = True):
    """Safely compile a function with fallback to CPU if TT compilation fails."""
    
    if device_manager.primary_device == "tt" and device_manager.is_tt_available():
        try:
            # Try to compile for TT device
            with device_manager.with_device("tt"):
                return jax.jit(fn)
        except Exception as e:
            logging.warning(f"TT compilation failed: {e}")
            if fallback_to_cpu:
                logging.info("Falling back to CPU compilation")
                with device_manager.with_device("cpu"):
                    return jax.jit(fn)
            else:
                raise
    else:
        # Compile for CPU
        with device_manager.with_device("cpu"):
            return jax.jit(fn)


def batch_operation_with_fallback(
    operation: Callable,
    data: jnp.ndarray,
    batch_size: int,
    device_manager: DeviceManager,
    fallback_batch_size: Optional[int] = None
) -> jnp.ndarray:
    """Perform batch operation with automatic fallback to smaller batches on CPU."""
    
    if fallback_batch_size is None:
        fallback_batch_size = batch_size // 2
    
    try:
        # Try with full batch size
        return operation(data)
    except Exception as e:
        logging.warning(f"Batch operation failed with batch size {batch_size}: {e}")
        
        if device_manager.primary_device == "tt":
            logging.info(f"Falling back to CPU with batch size {fallback_batch_size}")
            with device_manager.with_device("cpu"):
                # Process in smaller batches
                results = []
                for i in range(0, len(data), fallback_batch_size):
                    batch = data[i:i+fallback_batch_size]
                    batch_result = operation(batch)
                    results.append(batch_result)
                
                return jnp.concatenate(results, axis=0)
        else:
            raise


def memory_efficient_forward(
    model_fn: Callable,
    params: Any,
    inputs: jnp.ndarray,
    device_manager: DeviceManager,
    chunk_size: int = 1024
) -> jnp.ndarray:
    """Perform forward pass in chunks to manage memory efficiently."""
    
    if len(inputs) <= chunk_size:
        return model_fn(params, inputs)
    
    # Process in chunks
    results = []
    for i in range(0, len(inputs), chunk_size):
        chunk = inputs[i:i+chunk_size]
        
        try:
            with device_manager.with_device(device_manager.primary_device):
                chunk_result = model_fn(params, chunk)
        except Exception as e:
            logging.warning(f"Chunk processing failed on {device_manager.primary_device}: {e}")
            with device_manager.with_device("cpu"):
                chunk_result = model_fn(params, chunk)
        
        results.append(chunk_result)
    
    return jnp.concatenate(results, axis=0)


def create_device_manager(config) -> DeviceManager:
    """Create device manager from configuration."""
    return DeviceManager(
        primary_device=config.device_config.primary_device,
        fallback_device=config.device_config.fallback_device
    )


def log_device_info(device_manager: DeviceManager):
    """Log information about available devices."""
    logging.info(f"Primary device: {device_manager.primary_device}")
    logging.info(f"TT device available: {device_manager.is_tt_available()}")
    logging.info(f"Current device: {device_manager.current_device}")
    
    if device_manager.tt_device:
        logging.info(f"TT device: {device_manager.tt_device}")
    
    logging.info(f"CPU device: {device_manager.cpu_device}")
