# Getting Started w TT-Blacksmith

## Setup

To run experiments on Tenstorrent hardware, users must first activate correct environment.
On the first activation, script will automatically install all dependencies.

> Note:
> In case you cancel installation process or you want to install newer version
> it is recommended to:
> ```
> git pull
> rm -r ./env
> git restore env
> ```

### Activating frontend environment

To activate python environments for specific frontend, you need to run:

#### TT-Forge-FE:
```bash
source env/activate --ffe
```

#### TT-XLA:
```bash
source env/activate --xla
```

## Running Experiments

This section guides you through the process of running experiments included in this project, allowing you to reproduce results and explore different configurations.

- **Explore Available Experiments:** Browse the [experiments documentation](./experiments.md) to find a list of all available experiments.
- **Understand Experiment Details:** Before running an experiment, review its dedicated README file for high-level description and specific instructions.
- **Execute the Experiment:** Follow the detailed steps outlined in the experiment's README file to run it successfully.
- **Experiment with Configurations:** Feel free to modify the experiment configurations (e.g., parameters) as described in the README to observe their impact on the results.

## Visual Demo: 3D Reconstruction with NeRF

<img src="./imgs/nerf_demo.gif" alt="nerf demo" height="230"/>
