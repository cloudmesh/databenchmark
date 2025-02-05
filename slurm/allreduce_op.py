import torch
import torch.distributed as dist
import time, sys, os

def setup(rank, world_size):
    # Initialize the process group
    master_address = os.getenv("MASTER_ADDR")
    master_port = os.getenv("MASTER_PORT")
    
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_address}:{master_port}", 
        world_size=world_size,
        rank=rank
        )

def benchmark_bandwidth(size, iterations, device):
    data = torch.ones(size // 4, device=device, dtype=torch.float32)  # Convert bytes to floats

    # Warm-up
    for _ in range(10):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)

    # Benchmark
    start_time = time.time()
    for _ in range(iterations):
        dist.all_reduce(data, op=dist.ReduceOp.SUM, async_op=False)
    dist.barrier()
    elapsed_time = time.time() - start_time

    # Calculate bandwidth
    total_data = size * iterations * (dist.get_world_size() - 1)  # Total data communicated
    bandwidth = total_data / elapsed_time / (1024 ** 3)  # Convert to GB/s
    return bandwidth


## main
def main():

    iterations=100
    mega = 1024*1024 
    message_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

    # Environment variables set by SLURM
    rank = int(os.environ["SLURM_PROCID"])
    local_rank = int(os.environ["SLURM_LOCALID"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup(rank, world_size)
 
    # Assign each process to the GPU corresponding to its local rank
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    for size in message_sizes:
        bandwidth = benchmark_bandwidth(size*mega, iterations, device)
        if rank == 0:
            print(f'GPUs: {dist.get_world_size()}, size: {size} MByte, memory: {(torch.cuda.max_memory_allocated() / 1e9):.2f} GB, Bandwidth: {bandwidth:.4f} GB/s')

    # Cleanup
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
