#!/bin/bash
#SBATCH --job-name=torch-benchmark-distributed   # create a short name for your job
#SBATCH --nodes=2                # node count
#SBATCH --ntasks-per-node=1      # total number of tasks per node
#SBATCH --cpus-per-task=8       # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=300G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:4             # number of allocated gpus per node
#SBATCH --time=11:00:00       # total run time limit (HH:MM:SS)

##### Number of total processes 
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE
echo "XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX "

# export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
export MASTER_PORT=6000
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))
echo "WORLD_SIZE="$WORLD_SIZE

export MASTER_ADDR=$(scontrol show hostname ${SLURM_NODELIST} | head -n 1)
echo "MASTER_ADDR="$MASTER_ADDR
echo "$MASTER_ADDR:$MASTER_PORT"

export NCCL_SOCKET_IFNAME=ib
export NCCL_DEBUG=INFO

echo $SLURM_JOB_NUM_NODES
echo $SLURM_PROCID
echo $MASTER_ADDR
echo $MASTER_PORT

nnodes=$SLURM_JOB_NUM_NODES

module load mamba
micromamba activate ilab-benchmark-torch-env

launcher="python -u -m torch.distributed.run --nnodes=${nnodes} --master_addr ${MASTER_ADDR} --master_port ${MASTER_PORT} --nproc_per_node=4" 
echo $launcher 

cmd="train.py"
echo $cmd

echo "$launcher $cmd"

srun --jobid $SLURM_JOBID bash -c "$launcher --node_rank \$SLURM_PROCID $cmd" 

echo "END TIME: $(date)"
