#!/bin/bash
#SBATCH --job-name=swinv2-torch-benchmark-single-node   # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=1       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=300G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:1             # number of allocated gpus per node
#SBATCH --time=1:00:00       # total run time limit (HH:MM:SS)

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

export SLURM_CPUS_PER_TASK=1
export NCCL_SOCKET_IFNAME=ib
export NCCL_DEBUG=INFO

export PYTHONPATH=$PWD:$PWD/pytorch-caney

pip install timm yacs

cfg=swinv2_mim_pretrain_config.yaml
batch_size=128
num_samples=50000
dtype=$1

echo "Benchmark arguments"
echo "Batch-Size: ${batch_size}"
echo "Number of samples: ${num_samples}"
echo "DataType: ${dtype}"

cmd="python swinv2_benchmark.py --cfg ${cfg} --batch-size ${batch_size} --num_samples ${num_samples} --use-checkpoint --dtype ${dtype}"

echo $cmd

echo "START TIME: $(date)"

time $cmd 

echo "END TIME: $(date)"
