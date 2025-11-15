import torch
import numpy as np
import jax
from flax.core import unfreeze


def to_torch(x):
    """Converts a JAX array to a PyTorch tensor."""
    return torch.tensor(np.asarray(x))

def copy_jax_to_pt(params_jax, model_pt):
    
    params_jax = unfreeze(params_jax)['params']
    state_dict_pt = model_pt.state_dict()
    new_state_dict = {}

    new_state_dict['transformer.wte.weight'] = to_torch(params_jax['wte']['embedding'])
    new_state_dict['transformer.wpe.weight'] = to_torch(params_jax['wpe']['embedding'])
    new_state_dict['transformer.ln_f.weight'] = to_torch(params_jax['ln_f']['scale'])
    new_state_dict['transformer.ln_f.bias'] = to_torch(params_jax['ln_f']['bias'])
    
    new_state_dict['lm_head.weight'] = to_torch(params_jax['wte']['embedding'])

    num_layers = model_pt.config.n_layer
    for i in range(num_layers):
        jax_block = params_jax[str(i)]
        
        new_state_dict[f'transformer.h.{i}.ln_1.weight'] = to_torch(jax_block['ln_1']['scale'])
        new_state_dict[f'transformer.h.{i}.ln_1.bias'] = to_torch(jax_block['ln_1']['bias'])
        
        new_state_dict[f'transformer.h.{i}.attn.c_attn.weight'] = to_torch(jax_block['attn']['c_attn']['kernel'].T)
        new_state_dict[f'transformer.h.{i}.attn.c_attn.bias'] = to_torch(jax_block['attn']['c_attn']['bias'])
        new_state_dict[f'transformer.h.{i}.attn.c_proj.weight'] = to_torch(jax_block['attn']['c_proj']['kernel'].T)
        new_state_dict[f'transformer.h.{i}.attn.c_proj.bias'] = to_torch(jax_block['attn']['c_proj']['bias'])
        
        new_state_dict[f'transformer.h.{i}.ln_2.weight'] = to_torch(jax_block['ln_2']['scale'])
        new_state_dict[f'transformer.h.{i}.ln_2.bias'] = to_torch(jax_block['ln_2']['bias'])
        
        new_state_dict[f'transformer.h.{i}.mlp.c_fc.weight'] = to_torch(jax_block['mlp']['c_fc']['kernel'].T)
        new_state_dict[f'transformer.h.{i}.mlp.c_fc.bias'] = to_torch(jax_block['mlp']['c_fc']['bias'])
        new_state_dict[f'transformer.h.{i}.mlp.c_proj.weight'] = to_torch(jax_block['mlp']['c_proj']['kernel'].T)
        new_state_dict[f'transformer.h.{i}.mlp.c_proj.bias'] = to_torch(jax_block['mlp']['c_proj']['bias'])
    

    model_pt.load_state_dict(new_state_dict)
    print("✅ JAX weights successfully copied to PyTorch model.")

def compare_weights(wei_pt, wei_jax):

    def compare(t1, t2, name, rtol=1e-4, atol=1e-4):
        if not np.allclose(t1.cpu().numpy(), t2, rtol=rtol, atol=atol):
            miss.append(name)

    wei_jax = unfreeze(wei_jax)['params']
    miss = []

    compare(wei_pt['transformer.wte.weight'], wei_jax['wte']['embedding'], 'transformer.wte.weight')
    compare(wei_pt['transformer.wpe.weight'], wei_jax['wpe']['embedding'], 'transformer.wpe.weight')
    compare(wei_pt['transformer.ln_f.weight'], wei_jax['ln_f']['scale'], 'transformer.ln_f.weight')
    compare(wei_pt['transformer.ln_f.bias'], wei_jax['ln_f']['bias'], 'transformer.ln_f.bias')
    compare(wei_pt['lm_head.weight'], wei_jax['wte']['embedding'], 'lm_head.weight')
    num_layers = len([k for k in wei_jax.keys() if k.isdigit()])
    for i in range(num_layers):
        jax_block = wei_jax[str(i)]

        compare(wei_pt[f'transformer.h.{i}.ln_1.weight'], jax_block['ln_1']['scale'], f'transformer.h.{i}.ln_1.weight')
        compare(wei_pt[f'transformer.h.{i}.ln_1.bias'], jax_block['ln_1']['bias'], f'transformer.h.{i}.ln_1.bias')

        compare(wei_pt[f'transformer.h.{i}.attn.c_attn.weight'], jax_block['attn']['c_attn']['kernel'].T, f'transformer.h.{i}.attn.c_attn.weight')
        compare(wei_pt[f'transformer.h.{i}.attn.c_attn.bias'], jax_block['attn']['c_attn']['bias'], f'transformer.h.{i}.attn.c_attn.bias')
        compare(wei_pt[f'transformer.h.{i}.attn.c_proj.weight'], jax_block['attn']['c_proj']['kernel'].T, f'transformer.h.{i}.attn.c_proj.weight')
        compare(wei_pt[f'transformer.h.{i}.attn.c_proj.bias'], jax_block['attn']['c_proj']['bias'], f'transformer.h.{i}.attn.c_proj.bias')

        compare(wei_pt[f'transformer.h.{i}.ln_2.weight'], jax_block['ln_2']['scale'], f'transformer.h.{i}.ln_2.weight')
        compare(wei_pt[f'transformer.h.{i}.ln_2.bias'], jax_block['ln_2']['bias'], f'transformer.h.{i}.ln_2.bias')

        compare(wei_pt[f'transformer.h.{i}.mlp.c_fc.weight'], jax_block['mlp']['c_fc']['kernel'].T, f'transformer.h.{i}.mlp.c_fc.weight')
        compare(wei_pt[f'transformer.h.{i}.mlp.c_fc.bias'], jax_block['mlp']['c_fc']['bias'], f'transformer.h.{i}.mlp.c_fc.bias')
        compare(wei_pt[f'transformer.h.{i}.mlp.c_proj.weight'], jax_block['mlp']['c_proj']['kernel'].T, f'transformer.h.{i}.mlp.c_proj.weight')
        compare(wei_pt[f'transformer.h.{i}.mlp.c_proj.bias'], jax_block['mlp']['c_proj']['bias'], f'transformer.h.{i}.mlp.c_proj.bias')
        
    if len(miss) == 0:
        print("✅ SUCCESS: All weights match between JAX and PyTorch models!")
    else:
        print(f"❌ FAILURE: {len(miss)} parameters did not match.")
        print("Mismatched parameters:", miss)
