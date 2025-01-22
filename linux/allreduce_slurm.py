###### start of job script file #######
#!/bin/bash
#SBATCH --job-name=pytorch_distributed   # Job name
#SBATCH --mem=0                         # use all memory of the node (#SBATCH --mem=128G)
#SBATCH --nodes=2                       # Number of nodes (2)
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:4                    # Use all 4 GPUs per node
#SBATCH --ntasks=8                      # 8
#SBATCH --ntasks-per-node=4            # 4 tasks per node (1 per GPU)
    
# This is the number of GPUs that each tasks sees. There is strict 1 to 1 mapping, no contention.
#SBATCH --gpus-per-task=1               
#SBATCH --cpus-per-task=4               # Number of CPU cores per task (4)
#SBATCH --partition=a100                 # Partition name
#SBATCH --time=01:00:00                 # Time limit (1 hour)
#SBATCH --output=output_%j.log          # Standard output and error log
    
nvidia-smi -L
    
# Get the firts node fron the nodelist
MASTER_NODE_SHORT=$(scontrol show hostname $SLURM_NODELIST | head -n 1)
MASTER_ADDR=$(hostname -f)
MASTER_ADDR_IP=$(hostname -i)
#MASTER_PORT=29500
    
# Setting up environment
export MASTER_NODE_SHORT=$MASTER_NODE_SHORT
export MASTER_ADDR=$MASTER_ADDR
export MASTER_ADDR_IP=$MASTER_ADDR_IP
export MASTER_PORT=$MASTER_PORT
export WORLD_SIZE=$SLURM_NTASKS
# Global rank
export RANK=$SLURM_PROCID
    
# debugging
NCCL_DEBUG=INFO
NCCL_SOCKET_IFNAME=ib0  # Ensure InfiniBand is used
NCCL_IB_HCA=mlx5_0      # Specify InfiniBand device
    
#CUDA_VISIBLE_DEVICES=0,1,2,3
    
# Run the distributed script using srun
srun python allreduce_op.py
    
