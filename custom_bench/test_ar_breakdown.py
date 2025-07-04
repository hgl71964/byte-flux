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


def create_test_tensor(size: Tuple[int, ...], device: torch.device, rank: int, dtype=torch.float32) -> torch.tensor:
    """Create a test tensor with rank-specific values"""
    # Using a fixed seed that is different for each rank ensures reproducibility while having different data
    torch.manual_seed(rank)
    tensor = torch.randn(size, device=device, dtype=dtype)
    # Add rank-specific offset for easier verification
    tensor += rank * 0.1
    return tensor


def benchmark_all_reduce(tensor: torch.tensor, warmup_iters: int = 5, benchmark_iters: int = 100) -> Tuple[torch.tensor, float]:
    """Benchmark all-reduce operation"""
    # Warmup
    for _ in range(warmup_iters):
        temp_tensor = tensor.clone()
        if dist.is_initialized():
            dist.all_reduce(temp_tensor, op=dist.ReduceOp.SUM)
    
    # Get the correct result first
    result_tensor = tensor.clone()
    if dist.is_initialized():
        dist.all_reduce(result_tensor, op=dist.ReduceOp.SUM)
    
    # Benchmark timing only
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    start_time = time.time()
    
    for _ in range(benchmark_iters):
        temp_tensor = tensor.clone()
        if dist.is_initialized():
            dist.all_reduce(temp_tensor, op=dist.ReduceOp.SUM)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    end_time = time.time()
    avg_time = (end_time - start_time) / benchmark_iters
    
    return result_tensor, avg_time


def benchmark_reduce_scatter_all_gather(tensor: torch.tensor, warmup_iters: int = 5, benchmark_iters: int = 100, debug: bool = False) -> Tuple[torch.tensor, float]:
    """
    Benchmark reduce-scatter + all-gather operation using efficient tensor-based APIs.
    """
    if not dist.is_initialized():
        return tensor, 0.0

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    original_shape = tensor.shape
    
    # Use the more efficient tensor-based APIs: reduce_scatter_tensor and all_gather_into_tensor
    # These operate on contiguous tensors, not lists of chunks.
    tensor_flat = tensor.flatten()
    original_numel = tensor_flat.numel()
    
    # --- Padding to ensure divisibility by world_size ---
    pad_size = 0
    if original_numel % world_size != 0:
        pad_size = world_size - (original_numel % world_size)
        padding = torch.zeros(pad_size, device=tensor.device, dtype=tensor.dtype)
        tensor_flat = torch.cat([tensor_flat, padding])
    
    new_numel = tensor_flat.numel()
    chunk_size = new_numel // world_size
    
    # --- Prepare tensors for operations ---
    # For reduce_scatter_tensor, the output tensor is a chunk of the input size
    scattered_chunk = torch.empty(chunk_size, device=tensor.device, dtype=tensor.dtype)
    # For all_gather_into_tensor, the output tensor is the full size
    gathered_tensor_flat = torch.empty(new_numel, device=tensor.device, dtype=tensor.dtype)

    # --- Warmup ---
    for _ in range(warmup_iters):
        input_tensor_clone = tensor_flat.clone()
        dist.reduce_scatter_tensor(scattered_chunk, input_tensor_clone, op=dist.ReduceOp.SUM)
        dist.all_gather_into_tensor(gathered_tensor_flat, scattered_chunk)

    # --- Get the correct result once ---
    input_tensor_clone = tensor_flat.clone()
    dist.reduce_scatter_tensor(scattered_chunk, input_tensor_clone, op=dist.ReduceOp.SUM)
    dist.all_gather_into_tensor(gathered_tensor_flat, scattered_chunk)
    
    # Reconstruct final tensor for verification, removing padding if necessary
    if pad_size > 0:
        result_tensor = gathered_tensor_flat[:original_numel].reshape(original_shape)
    else:
        result_tensor = gathered_tensor_flat.reshape(original_shape)
    
    # --- Benchmark timing only ---
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time.time()
    
    for _ in range(benchmark_iters):
        input_tensor_clone = tensor_flat.clone()
        # Reduce-scatter
        dist.reduce_scatter_tensor(scattered_chunk, input_tensor_clone, op=dist.ReduceOp.SUM)
        # All-gather
        dist.all_gather_into_tensor(gathered_tensor_flat, scattered_chunk)
        
    if torch.cuda.is_available():
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
    return torch.allclose(tensor1, tensor2, rtol=rtol, atol=atol)


def calculate_bandwidth(tensor_bytes: int, latency_seconds: float, world_size: int) -> float:
    """
    Calculate algorithmic bandwidth in GB/s.
    For both All-Reduce and RS+AG, the total data moved across the network is
    proportional to 2 * (world_size - 1) / world_size * tensor_size.
    A common simplification for algorithmic bandwidth is to use 2 * tensor_size,
    representing a full send and receive of the data by each node.
    """
    if latency_seconds <= 0 or world_size <= 1:
        return 0.0
    
    # Algorithmic bandwidth calculation
    # Factor of 2 represents data in and data out
    total_bytes_per_process_algorithmic = 2 * ((world_size - 1) / world_size) * tensor_bytes
    
    # Convert to GB/s (1 GB = 1e9 bytes)
    bandwidth_gbps = total_bytes_per_process_algorithmic / (latency_seconds * 1e9)
    
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
        print(f"PyTorch Distributed Communication Benchmark (Corrected)")
        print(f"{'='*70}")
        print(f"World Size: {world_size}")
        print(f"Tensor Shape: {tensor_size}")
        print(f"Tensor Size: {tensor_bytes / 1024 / 1024:.2f} MB")
        print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
        print(f"{'='*70}")
        print(f"All-Reduce:")
        print(f"  Latency:    {all_reduce_time*1000:.3f} ms")
        print(f"  Bandwidth:  {all_reduce_bw:.2f} GB/s")
        print(f"Reduce-Scatter + All-Gather (Tensor API):")
        print(f"  Latency:    {reduce_scatter_gather_time*1000:.3f} ms")
        print(f"  Bandwidth:  {rs_ag_bw:.2f} GB/s")
        
        if world_size > 1 and reduce_scatter_gather_time > 0 and all_reduce_bw > 0:
            speedup = all_reduce_time / reduce_scatter_gather_time
            bw_ratio = rs_ag_bw / all_reduce_bw
            print(f"{'='*70}")
            print(f"Performance Comparison:")
            print(f"  Speedup (RS+AG vs All-Reduce): {speedup:.2f}x")
        
        print(f"{'='*70}")
        print(f"Results Match: {'✓ PASS' if results_match else '✗ FAIL'}")
        print(f"{'='*70}")


def main():
    parser = argparse.ArgumentParser(description='PyTorch Distributed Communication Benchmark')
    parser.add_argument('--tensor-size', type=int, nargs='+', default=[4096, 1024], 
                       help='Tensor dimensions (default: 4096 1024)')
    parser.add_argument('--warmup-iters', type=int, default=5, 
                       help='Number of warmup iterations (default: 5)')
    parser.add_argument('--benchmark-iters', type=int, default=100, 
                       help='Number of benchmark iterations (default: 100)')
    parser.add_argument('--rtol', type=float, default=1e-5, 
                       help='Relative tolerance for result verification (default: 1e-5)')
    parser.add_argument('--atol', type=float, default=1e-8, 
                       help='Absolute tolerance for result verification (default: 1e-8)')
    parser.add_argument('--debug', action='store_true', 
                       help='Enable debug output')
    
    args = parser.parse_args()
    
    # Setup distributed environment
    rank, world_size, device = setup_distributed()
    
    # Create test tensor
    tensor_size = tuple(args.tensor_size)
    test_tensor = create_test_tensor(tensor_size, device, rank)
    tensor_bytes = test_tensor.numel() * test_tensor.element_size()
    
    if rank == 0:
        print(f"Starting benchmark on {world_size} process(es)...")
        print(f"Tensor shape: {tensor_size}, Device: {device}")
    
    # Synchronize all processes before starting benchmarks
    if dist.is_initialized():
        dist.barrier()

    # Benchmark all-reduce
    all_reduce_result, all_reduce_time = benchmark_all_reduce(
        test_tensor, args.warmup_iters, args.benchmark_iters
    )
    
    # Benchmark reduce-scatter + all-gather
    rs_ag_result, rs_ag_time = benchmark_reduce_scatter_all_gather(
        test_tensor, args.warmup_iters, args.benchmark_iters, args.debug
    )
    
    # Verify results match
    if world_size > 1:
        # To be absolutely sure about logic, one could verify with float64
        # test_tensor_64 = create_test_tensor(tensor_size, device, rank, dtype=torch.float64)
        # ar_res_64, _ = benchmark_all_reduce(test_tensor_64, 5, 5)
        # rsag_res_64, _ = benchmark_reduce_scatter_all_gather(test_tensor_64, 5, 5)
        # assert torch.allclose(ar_res_64, rsag_res_64), "Logic failed on float64"
        
        results_match = verify_results(all_reduce_result, rs_ag_result, args.rtol, args.atol)
    else:
        results_match = True  # Single process case
    
    # Print results
    print_results(rank, world_size, tensor_size, all_reduce_time, rs_ag_time, 
                 results_match, tensor_bytes)
    
    # Cleanup
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()