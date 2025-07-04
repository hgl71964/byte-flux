import torch
import torch.distributed as dist
import time
import argparse
import os
from typing import Tuple, List

# A map from string names to torch.dtype objects
DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
}

def setup_distributed():
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
    else:
        # Default single-node setup
        rank = 0
        world_size = 1
        local_rank = 0
    
    # Initialize process group only if world_size is greater than 1
    if world_size > 1:
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            rank=rank,
            world_size=world_size
        )
    
    # Set device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    
    return rank, world_size, device


def create_test_tensor(size: Tuple[int, ...], device: torch.device, rank: int, dtype: torch.dtype) -> torch.Tensor:
    """Create a test tensor with rank-specific values and a given dtype."""
    tensor = torch.randn(size, device=device, dtype=dtype)
    # Add rank-specific offset for easier verification. Cast to tensor's dtype.
    tensor += torch.tensor(rank * 0.1, device=device, dtype=dtype)
    return tensor


def benchmark_all_reduce(tensor: torch.Tensor, warmup_iters: int = 5, benchmark_iters: int = 100) -> Tuple[torch.Tensor, float]:
    """Benchmark all-reduce operation."""
    # Warmup
    for _ in range(warmup_iters):
        temp_tensor = tensor.clone()
        dist.all_reduce(temp_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    
    # Get the correct result first for verification
    result_tensor = tensor.clone()
    dist.all_reduce(result_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    
    start_time = time.time()
    
    for _ in range(benchmark_iters):
        # CORRECTNESS FIX: Clone the tensor inside the loop.
        # This ensures we benchmark independent all-reduce operations,
        # not a sequence of reductions on an ever-increasing tensor value.
        temp_tensor = tensor.clone()
        dist.all_reduce(temp_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / benchmark_iters
    
    return result_tensor, avg_time


def benchmark_reduce_scatter_all_gather(tensor: torch.Tensor, warmup_iters: int = 5, benchmark_iters: int = 100) -> Tuple[torch.Tensor, float]:
    """Benchmark reduce-scatter + all-gather operation."""
    world_size = dist.get_world_size()
    original_shape = tensor.shape
    
    # Flatten tensor for processing
    tensor_flat = tensor.flatten()
    original_numel = tensor_flat.numel()
    
    # Pad tensor to be divisible by world_size if needed
    pad_size = 0
    if original_numel % world_size != 0:
        pad_size = world_size - (original_numel % world_size)
        padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor_flat = torch.cat([tensor_flat, padding])
    
    chunk_size = tensor_flat.numel() // world_size
    
    # Pre-allocate buffers for efficiency
    scattered_tensor = torch.zeros(chunk_size, device=tensor.device, dtype=tensor.dtype)
    gathered_tensors_list = [torch.zeros_like(scattered_tensor) for _ in range(world_size)]
    
    # Warmup
    for _ in range(warmup_iters):
        temp_tensor = tensor_flat.clone()
        dist.reduce_scatter(scattered_tensor, list(temp_tensor.chunk(world_size)), op=dist.ReduceOp.SUM)
        dist.all_gather(gathered_tensors_list, scattered_tensor)
    torch.cuda.synchronize()
    
    # Do one final operation to get the correct result for verification
    temp_result_tensor = tensor_flat.clone()
    dist.reduce_scatter(scattered_tensor, list(temp_result_tensor.chunk(world_size)), op=dist.ReduceOp.SUM)
    dist.all_gather(gathered_tensors_list, scattered_tensor)
    
    # Reconstruct final tensor from gathered chunks
    result_tensor = torch.cat(gathered_tensors_list)
    if pad_size > 0:
        result_tensor = result_tensor[:original_numel]
    result_tensor = result_tensor.reshape(original_shape)
    
    # Benchmark timing
    torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(benchmark_iters):
        # CORRECTNESS FIX: Clone the tensor inside the loop to ensure
        # each iteration benchmarks an independent operation.
        temp_tensor = tensor_flat.clone()
        input_list = list(temp_tensor.chunk(world_size))
        
        dist.reduce_scatter(scattered_tensor, input_list, op=dist.ReduceOp.SUM)
        dist.all_gather(gathered_tensors_list, scattered_tensor)
        
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time - start_time) / benchmark_iters
    
    return result_tensor, avg_time


def verify_results(tensor1: torch.Tensor, tensor2: torch.Tensor, rtol: float, atol: float) -> bool:
    """
    Verify that two tensors are approximately equal, using dtype-aware tolerances.
    If the user does not override the default tolerances, this function applies
    sensible defaults based on the tensor's precision.
    """
    effective_rtol, effective_atol = rtol, atol
    is_default_rtol = (rtol == 1e-5)
    is_default_atol = (atol == 1e-8)

    dtype = tensor1.dtype
    if dtype == torch.float16:
        if is_default_rtol: effective_rtol = 1e-2
        if is_default_atol: effective_atol = 1e-2
    elif dtype == torch.bfloat16:
        if is_default_rtol: effective_rtol = 1e-2
        if is_default_atol: effective_atol = 1e-2
    elif dtype == torch.float32:
        if is_default_rtol: effective_rtol = 1e-4
        if is_default_atol: effective_atol = 1e-5

    if dist.get_rank() == 0:
        print(f"\nVerifying results with dtype={dtype}, rtol={effective_rtol:.1e}, atol={effective_atol:.1e}")

    allclose = torch.allclose(tensor1, tensor2, rtol=effective_rtol, atol=effective_atol)
    
    if dist.get_rank() == 0:
        print("Verification FAILED: Tensors do not match.")
        diff = torch.abs(tensor1 - tensor2)
        max_diff = torch.max(diff)
        max_rel_diff = torch.max(diff / torch.abs(tensor1))
        # print(torch.abs(tensor1))
        # print(diff/torch.abs(tensor1))
        print(f"Max absolute difference: {max_diff.item():.3e}")
        print(f"Max relative difference: {max_rel_diff.item():.3e}")
    return allclose


def calculate_bandwidth(tensor_bytes: int, latency_seconds: float) -> float:
    """
    Calculate the algorithmic bandwidth in GB/s.
    For ring all-reduce and reduce-scatter + all-gather, the total data sent/received
    by each GPU is approximately 2 * (N-1)/N * tensor_size, which is ~2 * tensor_size for large N.
    Bandwidth = (2 * tensor_size_in_bytes) / time_in_seconds.
    """
    if latency_seconds <= 0:
        return 0.0
    
    # Algorithmic data transfer per process is 2 * tensor_size
    total_bytes_transferred = 2 * tensor_bytes
    
    # Convert to GB/s (1 GB = 1e9 bytes)
    bandwidth_gbps = total_bytes_transferred / (latency_seconds * 1e9)
    
    return bandwidth_gbps


def print_results(rank: int, world_size: int, tensor_size: Tuple[int, ...], 
                  dtype_str: str, all_reduce_time: float, 
                  reduce_scatter_gather_time: float, 
                  results_match: bool, tensor_bytes: int):
    """Print benchmark results on rank 0."""
    if rank == 0:
        all_reduce_bw = calculate_bandwidth(tensor_bytes, all_reduce_time)
        rs_ag_bw = calculate_bandwidth(tensor_bytes, reduce_scatter_gather_time)
        
        print(f"\n{'='*70}")
        print(f"PyTorch Distributed Communication Benchmark")
        print(f"{'='*70}")
        print(f"World Size:   {world_size}")
        print(f"Tensor Shape: {tensor_size}")
        print(f"Tensor Dtype: {dtype_str}")
        print(f"Tensor Size:  {tensor_bytes / 1024 / 1024:.2f} MB")
        print(f"Device:       {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"{'-'*70}")
        print(f"All-Reduce:")
        print(f"  Latency:    {all_reduce_time * 1000:.3f} ms")
        print(f"  Bandwidth:  {all_reduce_bw:.2f} GB/s")
        print(f"Reduce-Scatter + All-Gather:")
        print(f"  Latency:    {reduce_scatter_gather_time * 1000:.3f} ms")
        print(f"  Bandwidth:  {rs_ag_bw:.2f} GB/s")
        
        if world_size > 1:
            speedup = all_reduce_time / reduce_scatter_gather_time if reduce_scatter_gather_time > 0 else float('inf')
            bw_ratio = rs_ag_bw / all_reduce_bw if all_reduce_bw > 0 else float('inf')
            print(f"{'-'*70}")
            print(f"Performance Comparison:")
            print(f"  RS+AG is {speedup:.2f}x faster than All-Reduce.")
            print(f"  RS+AG achieves {bw_ratio:.2f}x the bandwidth of All-Reduce.")
        
        print(f"{'='*70}")
        print(f"Results Match: {'✅ PASS' if results_match else '❌ FAIL'}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='PyTorch Distributed Communication Benchmark')
    parser.add_argument('--tensor-size', type=int, nargs='+', default=[1024, 1024], 
                        help='Tensor dimensions (default: 1024 1024)')
    parser.add_argument('--dtype', type=str, default='float32', choices=DTYPE_MAP.keys(),
                        help=f'Data type for the tensor (default: float32)')
    parser.add_argument('--warmup-iters', type=int, default=10, 
                        help='Number of warmup iterations (default: 10)')
    parser.add_argument('--benchmark-iters', type=int, default=100, 
                        help='Number of benchmark iterations (default: 100)')
    parser.add_argument('--rtol', type=float, default=1e-5, 
                        help='Relative tolerance for result verification (default: 1e-5)')
    parser.add_argument('--atol', type=float, default=1e-8, 
                        help='Absolute tolerance for result verification (default: 1e-8)')
    
    args = parser.parse_args()
    
    rank, world_size, device = setup_distributed()
    
    selected_dtype = DTYPE_MAP[args.dtype]
    
    if 'cuda' in device.type:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is selected as device but not available.")
        if selected_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            if rank == 0:
                print(f"Warning: bfloat16 is not supported on this device. Skipping benchmark.")
            if world_size > 1: dist.destroy_process_group()
            return
    
    # Create test tensor
    tensor_size = tuple(args.tensor_size)
    test_tensor = create_test_tensor(tensor_size, device, rank, selected_dtype)
    tensor_bytes = test_tensor.numel() * test_tensor.element_size()
    
    if rank == 0:
        print(f"Starting benchmark on {world_size} process(es)...")
    
    # Benchmark all-reduce
    all_reduce_result, all_reduce_time = benchmark_all_reduce(
        test_tensor, args.warmup_iters, args.benchmark_iters
    )
    
    # Benchmark reduce-scatter + all-gather
    rs_ag_result, rs_ag_time = benchmark_reduce_scatter_all_gather(
        test_tensor, args.warmup_iters, args.benchmark_iters
    )
    
    # Verify results match
    if world_size > 1:
        results_match = verify_results(all_reduce_result, rs_ag_result, args.rtol, args.atol)
    else:
        results_match = True  # Single process case always matches
    
    # Print results
    print_results(rank, world_size, tensor_size, args.dtype, all_reduce_time, 
                  rs_ag_time, results_match, tensor_bytes)
    
    # Cleanup
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()