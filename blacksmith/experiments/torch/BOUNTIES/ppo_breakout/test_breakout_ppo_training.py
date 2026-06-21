def train_step():
    # Existing training code
    
    # Add cleanup after each step
    import torch_xla.core.xla_model as xm
    xm.mark_step()
    xm.wait_device_ops()
    
    # Explicitly clear TT-specific buffers
    if hasattr(torch, 'tt'):
        torch.tt.clear_caches()
        torch.tt.distributed.clear_host_buffers()