# Torch Distributed Benchmark and Test (Launcher)

## Purpose:
Benchmark training time of light-weight model and task using torch distributed backend with NCCL communication lib.

<i>Note: Uses distributed launcher</i>

## Customizations:
Edit the slurm script to make system-specific changes.

## How to run:

```bash
tar -xvf ../data/mnist_data.tar.gz 

sbatch multinode_torch_distributed_launcher_benchmark.sh
```