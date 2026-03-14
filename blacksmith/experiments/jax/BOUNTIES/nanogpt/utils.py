import torch
import numpy as np
import jax
from jax import numpy as jnp
import optax
from flax.core import unfreeze
from functools import partial

@partial(jax.jit, static_argnums=(2, 3), backend='tt')
def get_batch_native(data_dev, key, block_size, batch_size):
    # The hardware generates its own random indices natively
    ix = jax.random.randint(key, (batch_size,), 0, data_dev.shape[0] - block_size - 1)
    
    def extract_single_sequence(i):
        x = jax.lax.dynamic_slice(data_dev, (i,), (block_size,))
        y = jax.lax.dynamic_slice(data_dev, (i + 1,), (block_size,))
        return x, y

    # vmap vectorizes the extraction across the batch dimension instantly
    x_batch, y_batch = jax.vmap(extract_single_sequence)(ix)
    return x_batch, y_batch


@partial(jax.jit, static_argnums=(0, 1), backend='tt')
def train_step(model, optimizer, params, cache, opt_state, inputs, targets):
    ''' Standard training step with softmax cross-entropy loss. '''
    def loss_fn(p):
        variables = {'params': p['params'], **cache}
        logits = model.apply(variables, inputs, deterministic=True)
        
        vocab_size = logits.shape[-1]
        targets_oh = jax.nn.one_hot(targets, vocab_size)

        loss = optax.softmax_cross_entropy(logits, targets_oh)
        return jnp.mean(loss)

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss_val


def my_softmax(logits, labels):
    axis = -1
    label_logits = jnp.take_along_axis(logits, jnp.expand_dims(labels, axis), axis=axis).take(0, axis=axis)
    log_normalizers = jax.nn.logsumexp(logits, axis=axis)
    out = log_normalizers - label_logits
    return out


def to_torch(x):
    """Converts a JAX array to a PyTorch tensor."""
    return torch.tensor(np.asarray(x))

def copy_jax_to_pt(params_jax, model_pt):
    
    params_jax = unfreeze(params_jax)
    params_jax = params_jax['params'] | params_jax['cache']
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

    wei_jax = unfreeze(wei_jax)
    wei_jax = wei_jax['params'] | wei_jax['cache']
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
