import os
import torch
import torch.distributed as dist
import time
import numpy as np
import argparse
from datetime import timedelta
from functools import partial

# to run C++: https://github.com/nvidia/nccl-tests/issues/333

# Example torchrun commands:
# Single Op (e.g., all-gather on 2 nodes, 8 GPUs each):
# Node 0:
# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=2 \
#     --node_rank=0 \
#     --master_addr=<node0_ip> \
#     --master_port=29500 \
#     nccl_benchmark.py \
#     --op all-gather --iters 100 -r 0
# Node 1:
# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=2 \
#     --node_rank=1 \
#     --master_addr=<node0_ip> \
#     --master_port=29500 \
#     nccl_benchmark.py \
#     --op all-gather --iters 100 -r 0

# Run all Ops sequentially:
# Node 0:
# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=2 \
#     --node_rank=0 \
#     --master_addr=<node0_ip> \
#     --master_port=29500 \
#     nccl_benchmark.py \
#     --op all --iters 100 -r 0
# Node 1:
# torchrun \
#     --nproc_per_node=8 \
#     --nnodes=2 \
#     --node_rank=1 \
#     --master_addr=<node0_ip> \
#     --master_port=29500 \
#     nccl_benchmark.py \
#     --op all --iters 100 -r 0

# Constants
FLOAT32_BYTES = torch.finfo(torch.float32).bits // 8


def parse_args():
    parser = argparse.ArgumentParser(
        description="PyTorch Distributed Communication Bandwidth Measurement")
    parser.add_argument(
        '--op',
        type=str,
        choices=[
            'all-reduce',
            'all-gather',
            'reduce-scatter',
            'broadcast',
            'send-recv',  # Point-to-point ring
            'all'  # Run all supported tests sequentially
        ],
        required=True,
        help='Type of communication operation(s) to benchmark.')
    parser.add_argument('--iters',
                        type=int,
                        default=100,
                        help='Number of measurement iterations.')
    parser.add_argument('--warmup',
                        type=int,
                        default=20,
                        help='Number of warmup iterations.')
    parser.add_argument('-r',
                        type=int,
                        default=0,
                        help='Local rank responsible for printing results.')
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help='Random seed for tensor initialization.')
    parser.add_argument(
        '--timeout',
        type=int,
        default=60,
    )
    parser.add_argument(
        '--broadcast_src',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--allreduce_pool_size',
        type=int,
        default=10,
    )

    parser.add_argument(
        '--cap',
        type=float,
        default=1.5,
    )
    # Potentially add --sizes argument later if needed:
    # parser.add_argument('--sizes', type=str, default=None, help='Comma-separated list of sizes (e.g., "1KB,1MB,1GB")')
    parser.add_argument('--backend',
                        type=str,
                        default='nccl',
                        choices=['nccl', 'gloo'],
                        help='Distributed backend to use.')

    return parser.parse_args()


def format_size(size_in_bytes):
    """Convert bytes into a human-readable format (KB, MB, GB, etc.)."""
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(size_in_bytes)
    unit_index = 0
    while size >= 1024 and unit_index < len(units) - 1:
        size /= 1024.0
        unit_index += 1
    return f"{size:.2f} {units[unit_index]}"


def get_sizes(args):
    """Defines the tensor sizes (in bytes) to test."""
    sizes = [
        #1024,  # 1 KB
        1024 * 1024,  # 1 MB
        8 * 1024 * 1024,  # 8 MB
        64 * 1024 * 1024,  # 64 MB
        128 * 1024 * 1024,  # 128 MB
        512 * 1024 * 1024,  # 512 MB
        1024 * 1024 * 1024,  # 1 GB
        # Add larger sizes cautiously, ensure sufficient GPU memory
        2 * 1024 * 1024 * 1024,  # 2 GB
        4 * 1024 * 1024 * 1024,  # 4 GB
        8 * 1024 * 1024 * 1024,
        16 * 1024 * 1024 * 1024,
    ]
    # Allow overriding sizes via command line if needed in the future
    # if args.sizes:
    #     sizes = [parse_size_str(s) for s in args.sizes.split(',')]
    return sizes


def get_bandwidth_factors(op_name, world_size):
    """
    Get algorithm and bus bandwidth multipliers for different operations.
    For details, see:
        https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
    
    Returns:
        tuple: (algorithm_factor, bus_factor)
    """
    if op_name == 'all-reduce':
        # All-reduce: Each rank sends and receives data
        # Algorithm sees data size once (reduced result)
        # Bus sees 2(n-1)/n data volume (ring algorithm)
        algo_factor = 1.0
        bus_factor = 2.0 * (world_size - 1) / world_size
    elif op_name == 'reduce-scatter':
        # Reduce-scatter: Each rank contributes full data but receives 1/n
        # Algorithm sees 1/n of data (scattered result)
        # Bus sees (n-1)/n data volume (ring algorithm)
        algo_factor = world_size
        bus_factor = (world_size - 1) / world_size
    elif op_name == 'all-gather':
        # All-gather: Each rank contributes 1/n but receives full data
        # Algorithm sees full data size (gathered result)
        # Bus sees (n-1)/n data volume (ring algorithm)
        algo_factor = world_size
        bus_factor = (world_size - 1) / world_size
    elif op_name == 'broadcast':
        # Broadcast: Only one rank sends, all others receive
        # Algorithm sees full data size once
        # Bus factor depends on topology, approximated as (n-1)/n
        algo_factor = 1.0
        bus_factor = 1.0
    elif op_name == 'send-recv':
        # Point-to-point: Direct data transfer
        algo_factor = 1.0
        bus_factor = 1.0
    else:
        # Default conservative values
        # algo_factor = 1.0
        # bus_factor = 1.0
        raise RuntimeError(f'unsupport {op_name}')

    return algo_factor, bus_factor


def setup(
    timeout,
    backend='nccl',
):
    if dist.is_initialized():
        print(
            "Warning: Distributed process group already initialized. Skipping setup."
        )
        return

    # Ensure necessary env vars are set
    required_env_vars = [
        "RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"
    ]
    for var in required_env_vars:
        if var not in os.environ:
            raise RuntimeError(
                f"Required environment variable {var} is not set. "
                "Please launch using torchrun or similar.")

    rank = os.environ.get("RANK", "N/A")
    local_rank = os.environ.get("LOCAL_RANK", "N/A")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "None")

    timeout = timedelta(seconds=timeout)
    print(
        f"{rank=} {local_rank=} {visible_gpu=} {world_size=} Initializing process group with backend '{backend}' and timeout {timeout}..."
    )
    try:
        dist.init_process_group(
            backend=backend,
            timeout=timeout,
            # init_method='env://' is default and recommended
        )
        print(
            f"{rank=} {local_rank=} {visible_gpu=} {world_size=} Process group initialized successfully."
        )
        # Optional: Add a barrier here to ensure all processes initialized before proceeding
        dist.barrier()
        print(
            f"{rank=} {local_rank=} {visible_gpu=} {world_size=} Initial barrier passed."
        )
    except Exception as e:
        print(
            f"[Rank {rank}, LocalRank {local_rank}] {visible_gpu=} Error during init_process_group: {e}"
        )
        # Consider cleanup or re-raising depending on desired fault tolerance
        raise


def cleanup():
    """Destroy the distributed process group."""
    if dist.is_initialized():
        print("Cleaning up distributed process group...")
        dist.destroy_process_group()
        print("Process group destroyed.")
    else:
        print("Cleanup: Distributed process group not initialized.")


def print_test_header(op_name, rank, local_rank, world_size, visible_gpu):
    """Prints a standardized header for each test."""
    dist_rank = dist.get_rank() if dist.is_initialized() else rank
    dist_local_rank = local_rank  # Assuming local_rank derived correctly
    header = (
        f"--- Starting {op_name} Bandwidth Test --- \n"
        f"Global Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}\n"
        f"PyTorch Dist Rank: {dist_rank}, Visible GPUs: {visible_gpu}\n"
        f"PyTorch Dist Local Rank (derived): {dist_local_rank}")
    # Only print from the designated reporting rank (or rank 0 if not specified)
    reporting_rank = int(os.environ.get("REPORTING_RANK",
                                        0))  # Use env var if needed
    if rank == reporting_rank:
        print(header)
    dist.barrier()  # Ensure header is printed before tests start across ranks


# --- Measurement Functions ---

def measure_bandwidth(
    args,
    op_name,
    setup_tensors_func,
    comm_loop_func,
):
    """
    A unified, generic function to measure bandwidth for any communication pattern.

    Args:
        args: Command line arguments.
        op_name (str): Name of the operation (e.g., 'all-reduce').
        setup_tensors_func (callable): A function that creates and returns all
                                     necessary tensors for the operation.
                                     Signature: (size, device, world_size) -> tuple
        comm_loop_func (callable): A function that executes the core communication
                                 loop for a given number of iterations.
                                 Signature: (iters, comm_objects) -> None
    """
    try:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        visible_gpu = os.environ.get("CUDA_VISIBLE_DEVICES", "None")

        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')

        print_test_header(op_name, rank, local_rank, world_size, visible_gpu)

        sizes = get_sizes(args)
        algo_factor, bus_factor = get_bandwidth_factors(op_name, world_size)
        results = {}

        for size in sizes:
            if size > args.cap * 1024 * 1024 * 1024:
                continue

            # --- Tensor Creation and Cleanup ---
            comm_objects = None
            try:
                comm_objects = setup_tensors_func(size, device, world_size)
            except torch.cuda.OutOfMemoryError:
                if local_rank == args.r:
                    print(f"OOM ERROR: Cannot allocate tensor(s) for size {format_size(size)} on rank {rank}. Skipping.")
                dist.barrier()
                continue
            except Exception as e:
                if local_rank == args.r:
                    print(f"ERROR during tensor creation for size {format_size(size)} on rank {rank}: {e}")
                dist.barrier()
                raise

            # Ensure all ranks have allocated memory before starting
            dist.barrier()

            try:
                # --- Warmup ---
                if local_rank == args.r:
                    print(f"\n{rank=} Warming up for size {format_size(size)}...")
                comm_loop_func(args.warmup, comm_objects)
                torch.cuda.synchronize()
                dist.barrier()
                if local_rank == args.r:
                    print(f"{rank=} Warmup complete.")

                # --- Measurement ---
                if local_rank == args.r:
                    print(f"Starting measurement ({args.iters} iters) for size {format_size(size)}...")

                start_time = time.time()
                comm_loop_func(args.iters, comm_objects)
                torch.cuda.synchronize()
                end_time = time.time()
                dist.barrier()

                duration = end_time - start_time

                # --- Bandwidth Calculation ---
                if duration > 0:
                    total_data_algo = size * args.iters * algo_factor
                    algo_bandwidth_gbps = (total_data_algo / (1024**3)) / duration
                    bus_bandwidth_gbps = algo_bandwidth_gbps * bus_factor

                    if local_rank == args.r:
                        print(
                            f"  Size: {format_size(size)}\n"
                            f"    Total time: {duration:.4f} seconds\n"
                            f"    AlgoBW: {algo_bandwidth_gbps:.2f} GB/s - BusBW: {bus_bandwidth_gbps:.2f} GB/s"
                        )
                        results[size] = (algo_bandwidth_gbps, bus_bandwidth_gbps)
                else:
                     if local_rank == args.r:
                        print(f"  Size: {format_size(size)} - Invalid duration {duration:.4f}s. Skipping.")


            finally:
                # --- Cleanup per size ---
                del comm_objects
                torch.cuda.empty_cache()
                dist.barrier()

        if local_rank == args.r:
            print(f"--- {op_name} Test Complete ---")
        return results

    except Exception as e:
        local_rank = os.environ.get("LOCAL_RANK", "N/A")
        rank = os.environ.get("RANK", "N/A")
        print(f"[Rank {rank}, LocalRank {local_rank}] EXCEPTION in measure_bandwidth ({op_name}): {e}")
        import traceback
        traceback.print_exc()
        # Ensure other ranks don't hang if one rank errors out
        if dist.is_initialized():
            dist.barrier()


# --- Tensor and Communication Loop Setups ---
def setup_tensors_for_pooled_reducing_op(size, device, world_size, pool_size):
    assert pool_size > 0
    """
    Setup for in-place, reducing ops using a pool of buffers.
    This pre-allocates a number of tensors to be cycled through, avoiding
    the D2D copy overhead in the timing loop.
    """
    num_elements = size // FLOAT32_BYTES
    if num_elements == 0:
        raise ValueError("Size results in 0 elements")

    # This can consume significant memory: pool_size * size
    # A check could be added here for very large requests.
    tensors = [
        torch.rand(num_elements, dtype=torch.float32, device=device)
        for _ in range(pool_size)
    ]
    return (tensors,) # Return the pool as the first element of the tuple


def setup_single_tensor_for_overwrite(size, device, world_size):
    """
    Setup for ops that use a single tensor and overwrite it (e.g., broadcast).
    No pristine copy is needed as there's no accumulation.
    """
    num_elements = size // FLOAT32_BYTES
    if num_elements == 0: raise ValueError("Size results in 0 elements")
    tensor = torch.rand(num_elements, dtype=torch.float32, device=device)
    return (tensor,) # Return as a tuple

def setup_tensors_for_all_gather(size, device, world_size):
    """Setup for all-gather."""
    num_elements = size // FLOAT32_BYTES
    if num_elements == 0: raise ValueError("Size results in 0 elements")
    tensor = torch.rand(num_elements, dtype=torch.float32, device=device)
    output_list = [torch.empty_like(tensor) for _ in range(world_size)]
    return (tensor, output_list)

def setup_tensors_for_reduce_scatter(size, device, world_size):
    """Setup for reduce-scatter."""
    num_elements = size // FLOAT32_BYTES
    if num_elements == 0: raise ValueError("Size results in 0 elements")
    output_tensor = torch.empty(num_elements, dtype=torch.float32, device=device)
    input_list = [torch.rand_like(output_tensor) for _ in range(world_size)]
    return (output_tensor, input_list)

def setup_tensors_for_send_recv(size, device, world_size):
    """Setup for send-recv."""
    num_elements = size // FLOAT32_BYTES
    if num_elements == 0: raise ValueError("Size results in 0 elements")
    send_tensor = torch.rand(num_elements, dtype=torch.float32, device=device)
    recv_tensor = torch.empty_like(send_tensor)
    return (send_tensor, recv_tensor)




def comm_loop_in_place_pooled_reducing(comm_op, iters, comm_objects):
    """
    Communication loop for in-place, reducing ops using a buffer pool.
    It cycles through the pre-allocated buffers.
    """
    tensor_pool = comm_objects[0]
    pool_size = len(tensor_pool)
    for i in range(iters):
        # Use the next buffer in the pool, wrapping around.
        # This measures ONLY the communication op.
        comm_op(tensor_pool[i % pool_size])

def comm_loop_in_place_overwrite(comm_op, iters, comm_objects):
    """
    Communication loop for simple overwrite ops like broadcast.
    No reset is necessary.
    """
    tensor = comm_objects[0]
    for _ in range(iters):
        comm_op(tensor)

def comm_loop_list_output(comm_op, iters, comm_objects):
    """Communication loop for ops with a list output like all_gather."""
    tensor, output_list = comm_objects
    for _ in range(iters):
        comm_op(output_list, tensor)

def comm_loop_list_input(comm_op, iters, comm_objects):
    """Communication loop for ops with a list input like reduce_scatter."""
    output_tensor, input_list = comm_objects
    for _ in range(iters):
        comm_op(output_tensor, input_list)

def comm_loop_send_recv(iters, comm_objects):
    """Communication loop for the send-recv ring pattern."""
    send_tensor, recv_tensor = comm_objects
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Determine communication partners in a ring
    p_recv = (rank - 1 + world_size) % world_size
    p_send = (rank + 1) % world_size

    for _ in range(iters):
        # Even/odd pairing to prevent deadlock
        if rank % 2 == 0:
            dist.send(send_tensor, dst=p_send)
            dist.recv(recv_tensor, src=p_recv)
        else:
            dist.recv(recv_tensor, src=p_recv)
            dist.send(send_tensor, dst=p_send)

# --- Main Execution Logic ---

def main():
    args = parse_args()
    setup(args.timeout, args.backend)
    torch.manual_seed(args.seed)

    # --- Operation Registry ---
    # A data-driven way to define tests. Much cleaner than if/elifs.
    OP_REGISTRY = {
        'all-reduce': {
            # Use partial to pass the pool_size from args to the setup function
            'setup_func': partial(setup_tensors_for_pooled_reducing_op, pool_size=args.allreduce_pool_size),
            'comm_loop': partial(comm_loop_in_place_pooled_reducing, dist.all_reduce),
        },
        'broadcast': {
            'setup_func': setup_single_tensor_for_overwrite,
            'comm_loop': partial(comm_loop_in_place_overwrite, partial(dist.broadcast, src=args.broadcast_src)),
        },
        'all-gather': {
            'setup_func': setup_tensors_for_all_gather,
            'comm_loop': partial(comm_loop_list_output, dist.all_gather),
        },
        'reduce-scatter': {
            'setup_func': setup_tensors_for_reduce_scatter,
            'comm_loop': partial(comm_loop_list_input, dist.reduce_scatter),
        },
        'send-recv': {
            'setup_func': setup_tensors_for_send_recv,
            'comm_loop': comm_loop_send_recv,
        },
    }

    ops_to_run = OP_REGISTRY.keys() if args.op == 'all' else [args.op]
    all_results = {}

    for op_name in ops_to_run:
        if op_name not in OP_REGISTRY:
            if dist.get_rank() == 0:
                print(f"Unknown operation: {op_name}. Skipping.")
            continue

        config = OP_REGISTRY[op_name]
        results = measure_bandwidth(
            args,
            op_name=op_name,
            setup_tensors_func=config['setup_func'],
            comm_loop_func=config['comm_loop']
        )
        if results:
            all_results[op_name] = results

        # Small delay and barrier between different ops in 'all' mode
        time.sleep(0.5)
        dist.barrier()

    # Optional: Print summary if 'all' was run
    if args.op == 'all' and dist.get_rank() == args.r:
        print("\n\n===== All Operations Summary =====")
        for op_name, results in all_results.items():
            print(f"\n--- {op_name} Results ---")
            if not results:
                print("  No results recorded.")
                continue
            # Print first 5 and last 2 results for brevity
            keys = sorted(results.keys())
            items_to_print = keys[:5] + keys[-2:] if len(keys) > 7 else keys
            for size in items_to_print:
                bw = results[size]
                print(f"  Size: {format_size(size):>10s} -> Algo: {bw[0]:>6.2f} GB/s, Bus: {bw[1]:>6.2f} GB/s")
        print("================================\n")

    dist.barrier()
    cleanup()
    print("All ranks finished successfully.")

if __name__ == "__main__":
    main()