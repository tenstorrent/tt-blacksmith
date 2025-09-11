# Getting Started w TT-Blacksmith

## Setup

In order to make use of TT-Blacksmith experiements, you need to clone GitHub repository first:

```bash
git clone https://github.com/tenstorrent/tt-blacksmith.git
```

After cloning the repository you must build and activate one of the Python frontend environments (for TT-Forge-FE or TT-XLA) in order to make use of Tenstorrent hardware. In the following sections we will provide step by step instructions on how to build and activate your frontend environments.

### Build Frontend environment

#### TT-Forge-FE

To build the `TT-Forge-FE` frontend, you need to run:
```bash
./scripts/build_frontends.sh --ffe
```

#### TT-XLA

Since `TT-XLA` frontend depends on the MLIR environment, link to your toolchain directory needs to be stored in `TTMLIR_TOOLCHAIN_DIR`. If you do not specify toolchain directory, scripts will default to:
```
/opt/ttmlir-toolchain
```

If you're setting up for the first time (or don't have the MLIR environment installed), you need to perform a full build:
```bash
./scripts/build_frontends.sh --xla --full
```

For subsequent any builds, a regular rebuild is enough:
```bash
./scripts/build_frontends.sh --xla
```

---

### Activating Frontend Environment

To activate the previously built Python environments for specific frontends, you need to run:

#### TT-Forge-FE:
```bash
source ./scripts/activate_frontend.sh --ffe
```

#### TT-XLA:
```bash
source ./scripts/activate_frontend.sh --xla
```

### Cleaning Build Files

If at run into problems with building TT-XLA environment, try cleaning the previous build with:
 ```bash
source ./scripts/activate_frontend.sh --clean [--full]
```

Second parameter `--full` is optional and defines if you would like to remove TT-MLIR third party repository, so that the next build can start with fetching latest version of TT-MLIR and building it from scratch.

## Running Experiments

This section guides you through the process of running experiments included in this project, allowing you to reproduce results and explore different configurations.

- **Explore Available Experiments:** Browse the [experiments documentation](./experiments.md) to find a list of all available experiments.
- **Understand Experiment Details:** Before running an experiment, review its dedicated README file for high-level description and specific instructions.
- **Execute the Experiment:** Follow the detailed steps outlined in the experiment's README file to run it successfully.
- **Experiment with Configurations:** Feel free to modify the experiment configurations (e.g., parameters) as described in the README to observe their impact on the results.

## Visual Demo: 3D Reconstruction with NeRF

<img src="./imgs/nerf_demo.gif" alt="nerf demo" height="230"/>
