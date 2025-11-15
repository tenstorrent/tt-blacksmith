import jax

def main():
    # Check if JAX is using GPU or TPU
    backend = jax.default_backend()
    print(f"JAX is using the {backend} backend.")
    print(f"JAX version: {jax.__version__}")

if __name__ == "__main__":
    main()