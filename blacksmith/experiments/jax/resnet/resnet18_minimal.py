import jax
import jax.numpy as jnp
import flax.linen as nn

from blacksmith.tools.jax_utils import init_device

class ConvBatchNormBlock(nn.Module):
    filters: int
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        y = nn.Conv(self.filters, kernel_size=(3, 3), strides=(1, 1), 
                   padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        return y

def test_conv_batchnorm():
    model = ConvBatchNormBlock(filters=16)
    # Create random input: batch_size=2, height=32, width=32, channels=3
    with jax.default_device(jax.devices("cpu")[0]):
        key1, key2 = jax.random.split(jax.random.key(0))
        x_host = jax.random.normal(key1, (2, 32, 32, 5))
        variables_host = model.init(key2, x_host, train=True)
    
    x = jax.device_put(x_host, jax.devices("tt")[0])
    variables = jax.device_put(variables_host, jax.devices("tt")[0])
    # Create the model with 16 filters
    
    # Initialize parameters
    
    # Extract params and batch_stats
    params = variables['params']
    batch_stats = variables['batch_stats']
    
    # Forward function for computing gradients
    def forward_fn(params, batch_stats, inputs):
        outputs, new_state = model.apply(
            {'params': params, 'batch_stats': batch_stats},
            inputs,
            train=True,
            mutable=['batch_stats']
        )
        loss = jnp.mean(outputs)  # Simple loss: mean of outputs
        return loss, new_state
    
    # Compute gradients with respect to parameters
    grad_fn = jax.grad(lambda p, b, i: forward_fn(p, b, i)[0], argnums=0)
    grad_fn = jax.jit(grad_fn)
    grads = grad_fn(params, batch_stats, x)
    
    # Run the model in training mode
    output, updated_state = model.apply(variables, x, train=True, mutable=['batch_stats'])
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Gradient shapes:")
    for layer, layer_grads in grads.items():
        for param_name, param_grad in layer_grads.items():
            print(f"  {layer}/{param_name}: {param_grad.shape}")
    
    print("Test completed!")

# Run the test
if __name__ == "__main__":
    init_device()
    test_conv_batchnorm()