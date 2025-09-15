# Getting Started w TT-Blacksmith

## Setup

To run experiments on Tenstorrent hardware, users must first activate correct environment, on the first activation script will install all dependencies

> Note:
> In case you cancel installation process it is recommended to
> ```
> rm -r ./env
> git restore env
> ```

### Activating Frontend Environment

To activate the previously built Python environments for specific frontends, you need to run:

#### TT-Forge-FE:
```bash
source env/activate --ffe
```

#### TT-XLA:
```bash
source env/activate --xla
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
