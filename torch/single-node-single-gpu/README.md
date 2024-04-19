# Torch Benchmark and Test (Single Node Single GPU Single CPU)

## Purpose:
Benchmark training time of light-weight model and task using torch and CUDA.

## Customizations:
Edit the run script to make system-specific changes.

## How to run:
!!! Install the NVIDA NGC container
`singularity build --sandbox nvidia-pytorch-24.01 docker://nvcr.io/nvidia/pytorch:24.01-py3`
Note: container is already built on discover, prism

```bash
tar -xvf ../data/mnist_data.tar.gz 

chmod u+x multinode_torch_distributed_launcher_benchmark.sh 

# If using discover
salloc -N 1 --ntasks-per-node=1 --cpus-per-task=20 --time=2:00:00 --partition=gpu_a100 --constraint=rome --reservation=jcv --qos=8n_a100 -G 1

/discover/nobackup/jacaraba/spack/opt/spack/linux-sles15-zen/gcc-7.5.0/singularityce-3.11.3-o5pnooghlq7cgiv5zh5qnmyhmbltcynu/bin/singularity exec --nv -B /discover,/gpfsm /discover/nobackup/projects/akmosaic/container/nvpt-24.01 ./multinode_torch_distributed_launcher_benchmark.sh

# Elif using prism
salloc -G 1 --ntasks-per-node=1 --cpus-per-task=20 --time=2:00:00

module load singularity; singularity exec --nv -B /explore,/panfs /explore/nobackup/projects/ilab/containers/nvpt-24.01 ./multinode_torch_distributed_launcher_benchmark.sh

# Elif using grace-hopper
singularity exec --nv pytorch_24.01-py3.sif ./multinode_torch_distributed_launcher_benchmark.sh 
```