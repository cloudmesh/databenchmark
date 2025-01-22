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
     
    ###### end of job script file #######
     
     
    #### Start of Python program #####
    import torch
    import torch.distributed as dist
    import os, sys, math, time
     
    def setup(global_rank, world_size):
        # This is the number of GPUs that are visible for the task
        num_gpus = torch.cuda.device_count()
        device_id = torch.cuda.current_device()
        #print(f'GPUs seen by the task: {num_gpus}, device_id: {device_id}')
     
        # Check if SLURM_GPUS_PER_TASK is set and its value to 1
        num_gpus_is_set = os.environ.get('SLURM_GPUS_PER_TASK')
        if num_gpus_is_set is not None:
            num_gpus_seen_by_a_task = int(num_gpus_is_set)
            # Each tasks is mapped into a separate GPU. If gpus-per-task=1 then set device to 0
            if num_gpus_seen_by_a_task == 1:
                 local_rank = 0
                 torch.cuda.set_device(0)
            else:
                 local_rank = int(os.environ['SLURM_LOCALID'])
                 torch.cuda.set_device(local_rank)
        else:
            local_rank = int(os.environ['SLURM_LOCALID'])
            torch.cuda.set_device(local_rank)
     
        master_address = os.getenv("MASTER_ADDR")
        master_address_IP = os.getenv("MASTER_ADDR_IP")
        master_port = 29500 
     
        # Initialize the process group
        dist.init_process_group(
            backend="nccl",
            init_method=f"tcp://{master_address}:{master_port}",
            world_size=world_size,
            rank=global_rank
            )
        
        #print(f"Global Rank: {global_rank}, Local Rank: {local_rank}, Assigned GPU: {torch.cuda.current_device()}")
        return dist, local_rank
     
    def cleanup():
        dist.destroy_process_group()
     
    # Check if the batches are evenly disytributed
    def check_batch_allocation(dataloader, global_rank):
        total_tasks = int(os.environ.get("SLURM_NTASKS", 0))
        # Dataset and DataLoader
        dataset_size = len(dataloader.dataset)
        batch_size = dataloader.batch_size
     
        if(global_rank==0):
            # Compute total batches
            total_batches = math.ceil(dataset_size / batch_size)
     
            # Compute batches per GPU
            batches_per_gpu = sum(1 for _ in dataloader)
            calculated_samples = total_tasks*batches_per_gpu*batch_size
            print(f"number_of_samples:{dataset_size}, calculated_samples:{calculated_samples}, total_tasks: {total_tasks}, Batches per GPU: {batches_per_gpu}, batch_size:{batch_size}")
     
    #main 
    def main():
        torch.cuda.empty_cache()  # Clears GPU memory
        world_size = int(os.getenv("SLURM_NTASKS"))
        global_rank = int(os.environ['SLURM_PROCID'])  # Rank assigned by SLURM
        #local_rank = int(os.environ['SLURM_LOCALID'])  # Local rank
        print(f"start rank: {global_rank}")
     
        # Setup process group
        dist, local_rank = setup(global_rank, world_size)
        dist.barrier()
     
        rank = dist.get_rank()
        world_size = dist.get_world_size()
     
        # warmup, create a tensor for all_reduce
        size_in_bytes = 1024 * 1024 * 128  # 128 MB tensor
        tensor = torch.ones(size_in_bytes // 4, dtype=torch.float32)  # Float32 = 4 bytes
        tensor_cuda = tensor.cuda()
     
        # Warm-up (to avoid first-time overheads)
        for _ in range(10):
            dist.all_reduce(tensor_cuda, op=dist.ReduceOp.SUM)
     
        # Synchronize before timing
        dist.barrier()
     
        #tensor = torch.ones(10).cuda()
        mega = 1024*1024
        giga = mega*1024
        for i in range(13):  # Exponent from 0 to 10
            t_size = 2**i
            tensor = torch.rand(t_size*mega, dtype=torch.float32)
            # moving the tensor to GPU
            tensor_cuda = tensor.cuda()
            start = time.time()
            dist.barrier()
            dist.all_reduce(tensor_cuda, op=dist.ReduceOp.SUM, async_op=True)
            dist.barrier()
            exeTime = time.time() - start
            tensor_bytes = 4*t_size/1024 # Tensor size in GByte
            bandwidth = tensor_bytes/exeTime
            if global_rank == 0:
                print(f"Tensor: {tensor_bytes:.2f} GBytes, time: {exeTime:.2f}, bandwidth: {bandwidth:.2f} GByte/s")
     
        cleanup()
     
    # main   
    if __name__ == "__main__":
        try:
            main()
        except Exception as e:
            print(f"Error: {e}")

 
