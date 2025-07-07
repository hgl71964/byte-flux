# NVSHMEM_DISABLE_CUDA_VMM=1 torchrun --nproc_per_node=4 custom_bench/vllm_attn_moe_layer.py --tp_size 4

import os
from typing import Optional, Callable, Tuple, List
import numpy as np
import random

import argparse
import torch
import torch.distributed as dist

from vllm.distributed import (get_dp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_reduce_scatter,
                            get_tp_group,
                            init_world_group,
                              )
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)


def print_rank0(*args):
    if dist.get_rank() == 0:
        print(*args)

DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
    "s8": torch.int8,
    "s32": torch.int32,
}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--debug", action="store_true", help="debug mode. use human read input", default=False
    )
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--dtype", default="bfloat16", type=str, choices=list(DTYPE_MAP.keys()))
    args = parser.parse_args()
    return args

def init_seed(seed=0):
    torch.manual_seed(3 + seed)
    torch.cuda.manual_seed_all(3 + seed)
    np.random.seed(3 + seed)
    random.seed(3 + seed)

def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)
    init_seed(rank)
    return local_rank, rank, world_size, device


def test_allclose(tensor1, tensor2, threshold_map=None):
    if threshold_map is None:
        threshold_map = {
            torch.float16: 1e-2,
            torch.bfloat16: 2e-2,
            torch.float8_e4m3fn: 3e-2,
            torch.float8_e5m2: 3e-2,
            torch.int8: 2e-1,
        }
    
    # Ensure tensors have same dtype
    if tensor1.dtype != tensor2.dtype:
        raise ValueError(f"Tensor dtypes don't match: {tensor1.dtype} vs {tensor2.dtype}")
    
    # Get threshold for dtype, default to 1e-5 if not found
    threshold = threshold_map.get(tensor1.dtype, 1e-5)
    
    # Check if tensors are close
    is_close = torch.allclose(tensor1, tensor2, atol=threshold, rtol=threshold)
    
    # Calculate absolute difference between tensors
    abs_diff = torch.abs(tensor1 - tensor2)
    max_diff = abs_diff.max().item()
    
    # Calculate number of unmatched elements
    if not is_close:
        # Create boolean mask for elements that are NOT close
        close_mask = torch.isclose(tensor1, tensor2, atol=threshold, rtol=threshold)
        unmatched_count = (~close_mask).sum().item()
        total_elements = tensor1.numel()
    else:
        unmatched_count = 0
        total_elements = tensor1.numel()
    
    return {
        'is_close': is_close,
        'unmatched_count': unmatched_count,
        'total_elements': total_elements,
        'dtype': tensor1.dtype,
        'threshold': threshold,
        'max_diff': max_diff,
    }

def test_tensors(ref_out, test_out, name, results):
    if ref_out is not None and test_out is not None:
        results[name] = test_allclose(
            ref_out, test_out,
        )

def create_test_tensor(size: Tuple[int, ...], device: torch.device, rank: int, dtype: torch.dtype) -> torch.Tensor:
    # Create tensor in float32 for high-precision initial values.
    tensor_fp32 = torch.randn(size, device=device, dtype=torch.float32)
    # Add rank-specific offset for easier verification.
    tensor_fp32 += rank * 0.1
    # Cast to the target dtype.
    return tensor_fp32.to(dtype)

def main():
    args = parse_args()

    # TORCH distributed
    local_rank, rank, world_size, device = setup_distributed()
    assert args.tp_size == world_size
    print(f'world_size: {dist.get_world_size()}, tp_size: {args.tp_size}')
    print(f'rank: {dist.get_rank()} init on device: {device} and local_rank: {local_rank}')
    dist.barrier()

    ranks = list(range(torch.distributed.get_world_size()))
    import vllm.distributed.parallel_state as mpu
    mpu._WORLD = init_world_group(ranks, local_rank, 'nccl')
    ensure_model_parallel_initialized(
        args.tp_size,
        1,  # args.pp_size,
    )
    print(f'rank: {rank} init OK!')
    dist.barrier()

    # data
    dtype = DTYPE_MAP[args.dtype]
    r = {}
    partial_out = create_test_tensor((2048, 4096), device, rank, dtype)

    if args.debug:
        partial_out.zero_()
        partial_out[:, 0].fill_(rank + 1)
        print_rank0(f'partial_out: {partial_out}')
    dist.barrier()

    # AR - RS+AG
    ar_out = tensor_model_parallel_all_reduce(partial_out)
    rs_out = tensor_model_parallel_reduce_scatter(partial_out,0)
    ag_out = tensor_model_parallel_all_gather(rs_out,0)
    torch.cuda.synchronize()
    dist.barrier()
    print_rank0(f'ar_out: {ar_out.shape}, rs_out: {rs_out.shape}, ag_out: {ag_out.shape}')
    print_rank0(f'*'*20)
    test_tensors(ar_out, ag_out, 'out', r)

    if args.debug:
        print_rank0(f'ar_out: {ar_out}')
        print_rank0(f'rs_out: {rs_out}')
        print_rank0(f'ag_out: {ag_out}')
    dist.barrier()

    # higher precision for bloat16 (all good)
    if dtype == torch.bfloat16:
        ar_out_fp32 = tensor_model_parallel_all_reduce(partial_out.float())
        ar_out_accurate = ar_out_fp32.bfloat16()

        rs_out_fp32 = tensor_model_parallel_reduce_scatter(partial_out.float(), 0)
        ag_out_fp32 = tensor_model_parallel_all_gather(rs_out_fp32, 0)
        ag_out_accurate = ag_out_fp32.bfloat16()
        test_tensors(ar_out_accurate, ag_out_accurate, 'fp32-out', r)

    # print 
    for output_name, result in r.items():
        if 'error' in result:
            print_rank0(f"  {output_name}: ❌ FAIL - {result['error']}")
        else:
            status = "✅ PASS" if result['is_close'] else "❌ FAIL"
            dtype_str = str(result['dtype']).replace('torch.', '') if result['dtype'] else 'Unknown'
            
            if result['is_close']:
                print_rank0(f"  {output_name}: {status} "
                           f"(dtype: {dtype_str}, threshold: {result['threshold']:.2e}, "
                           f"MAX_DIFF: {result['max_diff']:.2e}"
                )
            else:
                unmatched_pct = (result['unmatched_count'] / result['total_elements']) * 100
                print_rank0(f"  {output_name}: {status} "
                           f"(dtype: {dtype_str}, threshold: {result['threshold']:.2e}, "
                           f"MAX_DIFF: {result['max_diff']:.2e}, "
                           f"unmatched: {result['unmatched_count']}/{result['total_elements']} "
                           f"({unmatched_pct:.2f}%))")
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
