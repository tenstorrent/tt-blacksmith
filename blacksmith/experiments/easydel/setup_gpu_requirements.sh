set -e

# TODO(pglusac): This is a temporary workaround. Refactor once we figure out what to do with torch-xla GPU package.
echo "Installing EasyDeL without deps (avoids triton conflict with torch)..."
pip install --no-deps \
    git+https://github.com/erfanzar/EasyDeL.git@77ced9d2f2ab6a3d705936d26112eb97d9f9e64a

echo "Installing EasyDeL dependencies"
pip install \
    "jax==0.7.1" \
    "jaxlib==0.7.1" \
    "flax==0.11.0" \
    "eformer>=0.0.62" \
    "einops~=0.8.0" \
    "optax>=0.2.2" \
    "jaxtyping~=0.3.2" \
    "ray[default]==2.34.0" \
    "fastapi>=0.115.2" \
    "uvloop==0.21.0" \
    "uvicorn[standard]>=0.32.0" \
    "jinja2>=3.1.5" \
    "grain~=0.2.11" \
    "datasets>=3.6.0" \
    "gcsfs>=2024.2,<2026" \
    "zstandard>=0.23.0" \
    "msgspec~=0.19.0" \
    "partial-json-parser>=0.2.1.1.post6" \
    "google-api-python-client>=2.179.0" \
    "cryptography>=45.0.6"
