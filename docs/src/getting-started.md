# Getting Started w TT-Blacksmith

## Setup

In order to make use of TT-Blacksmith experiements, you need to clone GitHub repository first:
```bash
git clone https://github.com/tenstorrent/tt-blacksmith.git
```

To run experiments on Tenstorrent hardware, users must activate correct environment.
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

To activate python environment for specific frontend (TT-XLA or TT-Forge-FE), you need to run:

```bash
source env/activate {--xla | --ffe}
```

## Running Experiments

This section guides you through the process of running experiments included in this project, allowing you to reproduce results and explore different configurations.

- **Explore Available Experiments:** Browse the [experiments documentation](./experiments.md) to find a list of all available experiments.
- **Understand Experiment Details:** Before running an experiment, review its dedicated README file for high-level description and specific instructions.
- **Execute the Experiment:** Follow the detailed steps outlined in the experiment's README file to run it successfully.
- **Experiment with Configurations:** Feel free to modify the experiment configurations (e.g., parameters) as described in the README to observe their impact on the results.

## Visual Demo: 3D Reconstruction with NeRF

<img src="./imgs/nerf_demo.gif" alt="nerf demo" height="230"/>
