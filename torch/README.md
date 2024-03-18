# PyTorch Benchmarking

This directory is dedicated to benchmarking code specific to PyTorch, tailored for both Nvidia-based and AMD-based GPU systems. Here you will find scripts, utilities, and resources designed to evaluate performance across different hardware configurations.

## Contents:

`mmulti-node-launcher`: Small benchmark to run and test torch distributed with NCCL using a torch launcher. See multi-node-launcher/README.md for more info

`multi-node-no-launcher`: Small benchmark to run and test torch distributed with NCCL using slurm env vars to initialize the distributed process. See multi-node-no-launcher/README.md for more info.

## Requirements:

`requirements/nvidia.pytorch.cuda.0.1.0.yaml`: Conda environment YAML file for Nvidia-based GPU systems.

`requirements/amd.pytorch.rocm.0.1.0.yaml`: Conda environment YAML file for AMD-based GPU systems.

## How to Use:

Clone this repository to your local machine:

```bash
git clone git@github.com:nasa-nccs-hpda/ilab-benchmarking-toolkit.git 
```
Navigate to the torch directory:

```bash
cd ilab-benchmarking-toolkit/torch
```

Choose the appropriate Conda environment YAML file based on your GPU system:


For Nvidia-based systems, create the environment:

```bash
micromamba create -f requirements/nvidia.pytorch.cuda.0.1.0.yaml
```

For AMD-based systems, create the environment:
```bash
micromamba create -f requirements/amd.pytorch.rocm.0.1.0.yaml
```

Each subdirectory contains a batch script which runs the corresponding bench script.
