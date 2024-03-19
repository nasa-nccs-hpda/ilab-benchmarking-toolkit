#!/bin/bash
#SBATCH -A geo160
#SBATCH --job-name=torch-benchmark-distributed   # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=100G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:8             # number of allocated gpus per node
#SBATCH --time=00:20:00       # total run time limit (HH:MM:SS)
#SBATCH -C nvme

echo "START TIME: $(date)"


##### Setup modules
module load cpe/23.05         # recommended cpe version with cray-mpich/8.1.26
module load cray-mpich/8.1.26 # for better GPU-aware MPI w/ ROCm 5.7.1
module load PrgEnv-gnu/8.4.0
export LD_LIBRARY_PATH=$CRAY_LD_LIBRARY_PATH:$LD_LIBRARY_PATH # because using a non-default cray-mpich
module load amd-mixed/5.7.1
module load craype-accel-amd-gfx90a
module load miniforge3/23.11.0
export MPICH_GPU_SUPPORT_ENABLED=1

##### sbcast env to local nvme
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

srun --ntasks-per-node 1 mkdir /mnt/bb/${USER}/${conda_env_name}
echo "untaring torchenv"
srun --ntasks-per-node 1 tar -xzf /mnt/bb/${USER}/${conda_env_name}.tar.gz -C /mnt/bb/${USER}/${conda_env_name}
echo "Done untarring torchenv"

source activate /mnt/bb/${USER}/${conda_env_name}
echo "Activated ${conda_env_name}"

srun --ntasks-per-node 1 conda-unpack

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# ******************* These are read internally it seems ***********************************
# ******** Master port, address and world size MUST be passed as variables for DDP to work 
export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "MASTER_PORT"=$MASTER_PORT
echo "WORLD_SIZE="$WORLD_SIZE

master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

# export NCCL_SOCKET_IFNAME=ib
export NCCL_DEBUG=INFO

# substitute with whatever code you are using
# just make sure srun is present
srun --unbuffered --gpu-bin=closest python train.py

echo "END TIME: $(date)"
