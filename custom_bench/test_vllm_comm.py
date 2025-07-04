# NVSHMEM_DISABLE_CUDA_VMM=1 torchrun --nproc_per_node=4 custom_bench/vllm_attn_moe_layer.py --tp_size 4

import os
from typing import Optional, Callable, Tuple, List
from copy import deepcopy
import numpy as np
import random

import argparse
import torch
import torch.distributed as dist

import triton


from vllm.engine.arg_utils import EngineArgs
from vllm.utils import FlexibleArgumentParser

from vllm.config import get_current_vllm_config
from vllm.platforms import current_platform
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.layers.fused_moe.layer import UnquantizedFusedMoEMethod

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.forward_context import ForwardContext, get_forward_context
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
from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm.model_executor.layers.linear import RowParallelLinear

import flux
from flux.cpp_mod import ReduceScatterOption


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
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=2048)

    parser.add_argument("--num_experts", type=int, default=128)
    parser.add_argument("--num_experts_per_tok", type=int, default=8,help='topK')
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--intermediate_size", type=int, default=4096)

    parser.add_argument("--params_dtype", type=str)
    parser.add_argument("--quant_config", type=str)

    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--ep_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", type=str, choices=list(DTYPE_MAP.keys()))

    rs = parser.add_argument_group('RS')
    rs.add_argument(
        "--transpose_weight", default=False, action="store_true", help="whether to transpose weight"
    )
    rs.add_argument(
        "--fuse_reduction", default=False, action="store_true", help="fuse reduction to gemm"
    )
    rs.add_argument(
        "--ring_reduction",
        default=False,
        action="store_true",
        help="reduce paritial output with ring order",
    )
    rs.add_argument("--has_bias", default=False, action="store_true", help="whether have bias")
    rs.add_argument(
        "--debug", action="store_true", help="debug mode. use human read input", default=False
    )
    rs.add_argument(
        "--use_1d_ring",
        action=argparse.BooleanOptionalAction,
        help="use 1d ring for reduction",
    )
    rs.add_argument(
        "--use_p2p_read",
        action=argparse.BooleanOptionalAction,
        help="use 1d ring for reduction",
    )
    rs.add_argument(
        "--use_cudaMemcpyAsync",
        action=argparse.BooleanOptionalAction,
        help="use 1d ring for reduction",
    )
    rs.add_argument(
        "--use_gemmk",
        action=argparse.BooleanOptionalAction,
        help="use 1d ring for reduction",
    )
    rs.add_argument(
        "--per_tile_flags",
        action=argparse.BooleanOptionalAction,
        help="use 1d ring for reduction",
    )
    rs.add_argument(
        "--reduce_scatter_blocks",
        type=int,
        help="number of blocks for reduce scatter",
    )
    rs.add_argument(
        "--ring_mode",
        choices=["ring1d", "ring2d"],
        help="ring mode. auto for auto detect",
    )
    args = parser.parse_args()
    return args

def init_seed(seed=0):
    os.environ["NCCL_DEBUG"] = os.getenv("NCCL_DEBUG", "ERROR")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_printoptions(precision=2)
    torch.manual_seed(3 + seed)
    torch.cuda.manual_seed_all(3 + seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    np.random.seed(3 + seed)
    random.seed(3 + seed)



def setup_distributed():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    device = torch.device(f'cuda:{local_rank}')
    torch.cuda.set_device(local_rank)

    # Set the device
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
    assert args.tp_size * args.ep_size == world_size, (f'world_size: {world_size}, tp_size: {args.tp_size}, ep_size: {args.ep_size}')
    print(f'world_size: {dist.get_world_size()}, tp_size: {args.tp_size}')
    print(f'rank: {dist.get_rank()} init on device: {device} and local_rank: {local_rank}')
    dist.barrier()

    local_rank = dist.get_rank()
    ranks = list(range(torch.distributed.get_world_size()))
    import vllm.distributed.parallel_state as mpu
    mpu._WORLD = init_world_group(ranks, local_rank, 'nccl')
    ensure_model_parallel_initialized(
        args.tp_size,
        1,  # args.pp_size,
    )

    # FLUX
    init_seed(rank)
    print_rank0("[flux_shm] before initialization")
    tp_group = get_tp_group()
    flux.init_flux_shm(tp_group.device_group)
    torch.cuda.synchronize()
    print_rank0("[flux_shm] after initialization")

    # from flux.testing import initialize_distributed
    # TP_GROUP = initialize_distributed()
    # RANK, WORLD_SIZE, NNODES = TP_GROUP.rank(), TP_GROUP.size(), flux.testing.NNODES()

    print(f'rank: {rank} init OK!')
    dist.barrier()

    # data
    dtype = DTYPE_MAP[args.dtype]
    # M = args.batch_size * args.seq_len
    # router_logits = torch.randn(M, args.num_experts, device=device, dtype=dtype)
    # seq_lens = [(args.seq_len, args.seq_len) for _ in range(args.batch_size)]
    # num_heads = (32, 32)
    # hidden_size_per_tp = args.hidden_size // args.tp_size
    # assert args.hidden_size % args.tp_size == 0 

    dist.barrier()

    r = {}
    partial_out = create_test_tensor((2048, 4096), device, rank, dtype)

    # AR - RS+AG
    ar_out = tensor_model_parallel_all_reduce(partial_out)

    rs_out = tensor_model_parallel_reduce_scatter(partial_out,0)
    ag_out = tensor_model_parallel_all_gather(rs_out,0)
    print_rank0(f'ar_out: {ar_out.shape}, rs_out: {rs_out.shape}, ag_out: {ag_out.shape}')
    test_tensors(ar_out, ag_out, 'out', r)

    # higher precision
    if dtype == torch.bfloat16:
        ar_out_fp32 = tensor_model_parallel_all_reduce(partial_out.float())
        ar_out_accurate = ar_out_fp32.bfloat16()

        rs_out_fp32 = tensor_model_parallel_reduce_scatter(partial_out.float(), 0)
        ag_out_fp32 = tensor_model_parallel_all_gather(rs_out_fp32, 0)
        ag_out_accurate = ag_out_fp32.bfloat16()
        test_tensors(ar_out_accurate, ag_out_accurate, 'fp32-out', r)

    # test:
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
