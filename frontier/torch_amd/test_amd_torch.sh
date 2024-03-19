#!/bin/bash
#SBATCH -A geo160
#SBATCH -J rocm_check
#SBATCH -t 00:10:00
#SBATCH -N 1
#SBATCH -C nvme

# Setup modules
module load cray-mpich/8.1.26 # for better GPU-aware MPI w/ ROCm 5.7.1
module load cpe/23.05 # recommended cpe version with cray-mpich/8.1.26
module load PrgEnv-gnu/8.4.0
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH # because using a non-default cray-mpich
module load amd-mixed/5.7.1
module load miniforge3/23.11.0
module load craype-accel-amd-gfx90a
export MPICH_GPU_SUPPORT_ENABLED=1

echo "copying torch_env to each node in the job"
conda_env_name='rocm-torch-test-full-0.1.0'

sbcast -pf $MEMBERWORK/geo160/${conda_env_name}.tar.gz /mnt/bb/${USER}/${conda_env_name}.tar.gz 
echo $MEMBERWORK/geo160/${conda_env_name}.tar.gz
echo /mnt/bb/${USER}/${conda_env_name}.tar.gz
ls -l /mnt/bb/${USER}
ls -l $MEMBERWORK/geo160

if [ ! "$?" == "0" ]; then
    # CHECK EXIT CODE. When SBCAST fails, it may leave partial files on the compute nodes, and if you continue to launch srun,
    # your application may pick up partially complete shared library files, which would give you confusing errors.
    echo "SBCAST failed!"
    exit 1
fi

srun -N 1 --ntasks-per-node 1 mkdir /mnt/bb/${USER}/${conda_env_name}
echo "untaring torchenv"
srun -N 1 --ntasks-per-node 1 tar -xzf /mnt/bb/${USER}/${conda_env_name}.tar.gz -C /mnt/bb/${USER}/${conda_env_name}
echo "Done untarring torchenv"

source activate /mnt/bb/${USER}/${conda_env_name}
echo "Activated ${conda_env_name}"

srun -N1 --ntasks-per-node 1 conda-unpack


srun --unbuffered -l -N 1 --gpus-per-node=1 --gpus-per-task=1 --gpu-bin=closest python test_amd_torch.py


