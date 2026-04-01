# Getting Started with TT-Blacksmith

This document walks you through how to set up TT-Blacksmith. TT-Blacksmith provides a collection of experiments for running machine learning workloads on Tenstorrent hardware.

> **NOTE:** If you encounter issues, please request assistance on the [TT-Blacksmith Issues](https://github.com/tenstorrent/tt-blacksmith/issues) page.

## Prerequisites

### 1. Set Up the Hardware

- Follow the instructions for the Tenstorrent device you are using at: [Hardware Setup](https://docs.tenstorrent.com/getting-started/README.html)

### 2. Install Software

- Follow the [manual software dependencies installation guide](https://docs.tenstorrent.com/getting-started/README.html) to set up your system.

## TT-Blacksmith Installation

This section walks through the installation steps for using a Docker container to run TT-Blacksmith experiments.

- **Prerequisite:** Docker must be installed. See the [official Docker installation guide](https://docs.docker.com/get-docker/) if needed.

### Step 1. Run the Docker Container

```bash
docker run -it --rm \
  --device /dev/tenstorrent \
  -v /dev/hugepages-1G:/dev/hugepages-1G \
  ghcr.io/tenstorrent/tt-xla/tt-xla-base-ubuntu-24-04:latest
```

> **NOTE:** You cannot isolate devices in containers. You must pass through all devices even if you are only using one. You can do this by passing `--device /dev/tenstorrent`. Do not try to pass `--device /dev/tenstorrent/1` or similar, as this type of device-in-container isolation will result in fatal errors later on during execution.

- If you want to check that it is running, open a new tab with the **Same Command** option and run the following:

```bash
docker ps
```

### Step 2. Running Experiments in Docker

Inside your running Docker container:

1. Clone the TT-Blacksmith repo:

```bash
git clone https://github.com/tenstorrent/tt-blacksmith.git
cd tt-blacksmith
```

2. Activate the environment:

```bash
source env/activate --xla
```

> **NOTE:** To run experiments on GPU instead of Tenstorrent hardware, Docker is not required. Simply clone the repository and use `source env/activate --gpu`.

3. Run an experiment by following the instructions in its README file. For example, to run the LLaMA LoRA fine-tuning experiment, see the [LoRA README](https://github.com/tenstorrent/tt-blacksmith/blob/main/blacksmith/experiments/torch/llama/xla/lora/README.md).

---

## Running Experiments

This section guides you through the process of running experiments included in this project, allowing you to reproduce results and explore different configurations.

- **Explore Available Experiments:** Browse the [experiments documentation](./experiments.md) to find a list of all available experiments.
- **Understand Experiment Details:** Before running an experiment, review its dedicated README file for high-level description and specific instructions.
- **Execute the Experiment:** Follow the detailed steps outlined in the experiment's README file to run it successfully.
- **Experiment with Configurations:** Feel free to modify the experiment configurations (e.g., parameters) as described in the README to observe their impact on the results.

## Where to Go Next

- Explore more experiments in the [experiments documentation](./experiments.md)
- Check out the [TT-XLA documentation](https://docs.tenstorrent.com/tt-xla/) for more information about the underlying framework
- Use [TT-Installer](https://docs.tenstorrent.com/getting-started/README.html) for a quick software installation path

## Visual Demo: 3D Reconstruction with NeRF

<img src="https://raw.githubusercontent.com/tenstorrent/tt-blacksmith/main/docs/shared/images/nerf_demo.gif" alt="nerf demo" height="230"/>
