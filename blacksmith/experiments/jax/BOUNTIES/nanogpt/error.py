import jax.numpy as jnp
import optax
import jax

def train_step(model, optimizer, params, cache, opt_state, inputs, targets):
    ''' Standard training step with softmax cross-entropy loss. '''
    def loss_fn(p):
        variables = {'params': p['params'], **cache}
        logits = model.apply(variables, inputs, deterministic=True)

        label_logits = jnp.take_along_axis(logits, jnp.expand_dims(targets, -1), axis=-1).take(0, axis=-1)
        log_normalizers = jax.nn.logsumexp(logits, axis=-1)
        loss = log_normalizers - label_logits
        return jnp.mean(loss)

    loss_val, grads = jax.value_and_grad(loss_fn)(params)
    
    updates, new_opt_state = optimizer.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    
    return new_params, new_opt_state, loss_val


def my_softmax(logits, labels):
    label_logits = jnp.take_along_axis(logits, jnp.expand_dims(labels, -1), axis=-1).take(0, axis=-1)
    log_normalizers = jax.nn.logsumexp(logits, axis=-1)
    out = log_normalizers - label_logits
    return out

