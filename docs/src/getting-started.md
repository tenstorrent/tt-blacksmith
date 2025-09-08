# Getting Started w TT-Blacksmith

## Setup

To run experiments on Tenstorrent hardware, users must first build and activate either the TT-Forge-FE (for PyTorch) or TT-XLA (for JAX) frontend environment using the provided scripts.

### Build Frontend environment

#### TT-Forge-FE

To build the `TT-Forge-FE` frontend, run:
```bash
./scripts/build_frontends.sh --ffe
```

#### TT-XLA

Since `TT-XLA` depends on the MLIR environment, you can set the `TTMLIR_TOOLCHAIN_DIR` to point to your toolchain directory. If not specified, it defaults to:
```
/opt/ttmlir-toolchain
```

If you're setting up for the first time (or don't have the MLIR environment installed), do a full build:
```bash
./scripts/build_frontends.sh --xla --full
```

For subsequent builds, a regular rebuild is enough:
```bash
./scripts/build_frontends.sh --xla
```

---

### Activating Frontend Environment

To activate the Python environment for a specific frontend:

For `TT-Forge-FE`:
```bash
source ./scripts/activate_frontend.sh --ffe
```

For `TT-XLA`:
```bash
source ./scripts/activate_frontend.sh --xla
```

## Running Experiments

This section guides you through the process of running experiments included in this project, allowing you to reproduce results and explore different configurations.

- **Explore Available Experiments:** Browse the [experiments documentation](./experiments.md) to find a list of all available experiments.
- **Understand Experiment Details:** Before running an experiment, review its dedicated README file for high-level description and specific instructions.
- **Execute the Experiment:** Follow the detailed steps outlined in the experiment's README file to run it successfully.
- **Experiment with Configurations:** Feel free to modify the experiment configurations (e.g., parameters) as described in the README to observe their impact on the results.

## Visual Demo: 3D Reconstruction with NeRF

<img src="imgs/nerf_demo.gif" alt="nerf demo" height="230"/>
