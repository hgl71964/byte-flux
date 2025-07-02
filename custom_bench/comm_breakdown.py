import torch
import torch.distributed as dist
import time
import argparse
import os
from typing import Tuple, List


def setup_distributed():
    """Initialize distributed training environment"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    else:
        # Default single-node setup
        rank = 0
        world_size = 1
        local_rank = 0
    
    # Initialize process group
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


def create_test_tensor(size: Tuple[int, ...], device: torch.device, rank: int) -> torch.tensor:
    """Create a test tensor with rank-specific values"""
    tensor = torch.randn(size, device=device, dtype=torch.float32)
    # Add rank-specific offset for easier verification
    tensor += rank * 0.1
    return tensor


def benchmark_all_reduce(tensor: torch.tensor, warmup_iters: int = 5, benchmark_iters: int = 100) -> Tuple[torch.tensor, float]:
    """Benchmark all-reduce operation"""
    # Warmup
    for _ in range(warmup_iters):
        temp_tensor = tensor.clone()
        dist.all_reduce(temp_tensor, op=dist.ReduceOp.SUM)
    
    # Get the correct result first
    result_tensor = tensor.clone()
    dist.all_reduce(result_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    
    start_time = time.time()
    
    temp_tensor = tensor.clone()
    for _ in range(benchmark_iters):
        dist.all_reduce(temp_tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()

    end_time = time.time()
    avg_time = (end_time - start_time) / benchmark_iters
    
    return result_tensor, avg_time


def benchmark_reduce_scatter_all_gather(tensor: torch.tensor, warmup_iters: int = 5, benchmark_iters: int = 100) -> Tuple[torch.tensor, float]:
    """Benchmark reduce-scatter + all-gather operation"""
    
    world_size = dist.get_world_size()
    original_shape = tensor.shape
    
    # Flatten tensor for reduce-scatter
    tensor_flat = tensor.flatten()
    original_numel = tensor_flat.numel()
    
    # Pad tensor to make it divisible by world_size if needed
    pad_size = 0
    if tensor_flat.numel() % world_size != 0:
        pad_size = world_size - (tensor_flat.numel() % world_size)
        padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor_flat = torch.cat([tensor_flat, padding])
    
    chunk_size = tensor_flat.numel() // world_size
    if dist.get_rank() == 0:
        print(f'pad_size: {pad_size}, chunk_size: {chunk_size}')
    
    # Warmup
    for _ in range(warmup_iters):
        temp_tensor = tensor_flat.clone()
        # Reduce-scatter
        scattered_tensor = torch.zeros(chunk_size, device=tensor.device, dtype=tensor.dtype)
        dist.reduce_scatter(scattered_tensor, list(temp_tensor.chunk(world_size)), op=dist.ReduceOp.SUM)
        # All-gather
        gathered_tensors = [torch.zeros_like(scattered_tensor) for _ in range(world_size)]
        dist.all_gather(gathered_tensors, scattered_tensor)
    
    # Do one final operation to get the correct result
    temp_tensor = tensor_flat.clone()
    scattered_tensor = torch.zeros(chunk_size, device=tensor.device, dtype=tensor.dtype)
    dist.reduce_scatter(scattered_tensor, list(temp_tensor.chunk(world_size)), op=dist.ReduceOp.SUM)
    gathered_tensors = [torch.zeros_like(scattered_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, scattered_tensor)
    
    # Reconstruct final tensor for verification
    result_tensor = torch.cat(gathered_tensors)
    if pad_size > 0:
        result_tensor = result_tensor[:original_numel]
    result_tensor = result_tensor.reshape(original_shape)
    
    # Now benchmark timing only
    torch.cuda.synchronize()
    
    start_time = time.time()
    
    temp_tensor = tensor_flat.clone()
    partial_tensor = list(temp_tensor.chunk(world_size))
    scattered_tensor = torch.zeros(chunk_size, device=tensor.device, dtype=tensor.dtype)
    gathered_tensors_bench = [torch.zeros_like(scattered_tensor) for _ in range(world_size)]
    for _ in range(benchmark_iters):
        # Reduce-scatter
        dist.reduce_scatter(scattered_tensor, partial_tensor, op=dist.ReduceOp.SUM)
        
        # All-gather
        dist.all_gather(gathered_tensors_bench, scattered_tensor)
        
    torch.cuda.synchronize()
    end_time = time.time()
    avg_time = (end_time - start_time) / benchmark_iters
    
    return result_tensor, avg_time


def verify_results(tensor1: torch.tensor, tensor2: torch.tensor, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    """Verify that two tensors are approximately equal"""
    if tensor1.dtype == torch.float32:
        # For float32, relax the tolerance slightly for large reductions
        # due to non-associativity of floating-point addition.
        rtol = max(rtol, 1e-4)
        atol = max(atol, 1e-5)
    allclose =  torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)
    if not allclose:
        if dist.get_rank() == 0:
            not_close_mask = ~torch.isclose(tensor1, tensor2, rtol=rtol, atol=atol)
            num_not_close = not_close_mask.sum().item()
            print(f"Number of non-matching elements: {num_not_close}")
            indices = torch.nonzero(not_close_mask, as_tuple=False)
            print("Indices of differing elements:", indices)

        # Optionally print the actual values
        # for idx in indices[:10]:  # show only first 10 mismatches for brevity
        #     i = tuple(idx.tolist())
        #     print(f"tensor1{ i } = { tensor1[i] }, tensor2{ i } = { tensor2[i] }")

    return allclose


def calculate_bandwidth(tensor_bytes: int, latency_seconds: float, world_size: int) -> float:
    """
    Calculate bandwidth in GB/s for collective communication operations.
    
    For all-reduce: each process sends and receives (world_size - 1) * tensor_size bytes
    For reduce-scatter + all-gather: 
        - reduce-scatter: each process sends tensor_size bytes, receives tensor_size/world_size bytes
        - all-gather: each process sends tensor_size/world_size bytes, receives tensor_size bytes
        - Total: 2 * tensor_size bytes per process
    
    Args:
        tensor_bytes: Size of tensor in bytes
        latency_seconds: Time taken for the operation in seconds
        world_size: Number of processes
    
    Returns:
        Bandwidth in GB/s
    """
    if latency_seconds <= 0:
        return 0.0
    
    # For collective operations, we consider the total data movement per process
    # All-reduce effectively moves (world_size - 1) * tensor_size bytes per process
    # But for bandwidth calculation, we typically use the algorithmic bandwidth
    # which is 2 * tensor_size for both operations (send + receive)
    total_bytes = 2 * tensor_bytes
    
    # Convert to GB/s (using 1 GB = 1e9 bytes for network bandwidth convention)
    bandwidth_gbps = total_bytes / (latency_seconds * 1e9)
    
    return bandwidth_gbps


def print_results(rank: int, world_size: int, tensor_size: Tuple[int, ...], 
                 all_reduce_time: float, reduce_scatter_gather_time: float, 
                 results_match: bool, tensor_bytes: int):
    """Print benchmark results"""
    if rank == 0:
        # Calculate bandwidths
        all_reduce_bw = calculate_bandwidth(tensor_bytes, all_reduce_time, world_size)
        rs_ag_bw = calculate_bandwidth(tensor_bytes, reduce_scatter_gather_time, world_size)
        
        print(f"\n{'='*70}")
        print(f"PyTorch Distributed Communication Benchmark")
        print(f"{'='*70}")
        print(f"World Size: {world_size}")
        print(f"Tensor Shape: {tensor_size}")
        print(f"Tensor Size: {tensor_bytes / 1024 / 1024:.2f} MB")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"{'='*70}")
        print(f"All-Reduce:")
        print(f"  Latency:    {all_reduce_time*1000:.3f} ms")
        print(f"  Bandwidth:  {all_reduce_bw:.2f} GB/s")
        print(f"Reduce-Scatter + All-Gather:")
        print(f"  Latency:    {reduce_scatter_gather_time*1000:.3f} ms")
        print(f"  Bandwidth:  {rs_ag_bw:.2f} GB/s")
        
        if world_size > 1:
            speedup = all_reduce_time / reduce_scatter_gather_time if reduce_scatter_gather_time > 0 else float('inf')
            bw_ratio = rs_ag_bw / all_reduce_bw if all_reduce_bw > 0 else float('inf')
            print(f"{'='*70}")
            print(f"Performance Comparison:")
            print(f"  Latency Ratio (All-Reduce/RS+AG): {speedup:.2f}x")
            print(f"  Bandwidth Ratio (RS+AG/All-Reduce): {bw_ratio:.2f}x")
        
        print(f"{'='*70}")
        print(f"Results Match: {'✅ PASS' if results_match else '❌ FAIL'}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='PyTorch Distributed Communication Benchmark')
    parser.add_argument('--tensor-size', type=int, nargs='+', default=[1024, 1024], 
                       help='Tensor dimensions (default: 1024 1024)')
    parser.add_argument('--warmup-iters', type=int, default=5, 
                       help='Number of warmup iterations (default: 5)')
    parser.add_argument('--benchmark-iters', type=int, default=100, 
                       help='Number of benchmark iterations (default: 100)')
    parser.add_argument('--rtol', type=float, default=1e-5, 
                       help='Relative tolerance for result verification (default: 1e-5)')
    parser.add_argument('--atol', type=float, default=1e-8, 
                       help='Absolute tolerance for result verification (default: 1e-8)')
    
    args = parser.parse_args()
    
    # Setup distributed environment
    rank, world_size, device = setup_distributed()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    if not dist.is_initialized():
        raise RuntimeError("Distributed environment not initialized") 
    
    # Create test tensor
    tensor_size = tuple(args.tensor_size)
    test_tensor = create_test_tensor(tensor_size, device, rank)
    tensor_bytes = test_tensor.numel() * test_tensor.element_size()
    
    if rank == 0:
        print(f"Starting benchmark on {world_size} process(es)...")
        print(f"Tensor shape: {tensor_size}, Device: {device}")
    
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
        results_match = True  # Single process case
    
    # Print results
    print_results(rank, world_size, tensor_size, all_reduce_time, rs_ag_time, 
                 results_match, tensor_bytes)
    
    # Cleanup
    dist.destroy_process_group()


if __name__ == "__main__":
    main()