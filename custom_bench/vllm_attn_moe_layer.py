# NVSHMEM_DISABLE_CUDA_VMM=1 torchrun --nproc_per_node=4 custom_bench/vllm_attn_moe_layer.py --tp_size 4

# NOTE: NCCL accumulates at bf16 introduces numerical issue
# we can use stable reduction algo - eliminates the numerical issue - however FLUX does not support that
# NCCL_ALGO=tree,NVLSTree,NVLS  NVSHMEM_DISABLE_CUDA_VMM=1 torchrun --nproc_per_node=4 custom_bench/vllm_attn_moe_layer.py --tp_size 4

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
from vllm.model_executor.layers.fused_moe.fused_moe import fused_topk
from vllm.triton_utils import tl
from vllm.model_executor.layers.fused_moe.moe_align_block_size import moe_align_block_size
from vllm import _custom_ops as ops

from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.distributed import (get_dp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce,
                              tensor_model_parallel_all_gather,
                              tensor_model_parallel_reduce_scatter,
                            get_tp_group, get_ep_group, get_pp_group,
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
    parser.add_argument("--topk", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=4096)
    parser.add_argument("--intermediate_size", type=int, default=4096)

    parser.add_argument("--params_dtype", type=str)
    parser.add_argument("--quant_config", type=str)

    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--ep_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", type=str, choices=list(DTYPE_MAP.keys()))
    parser.add_argument("--quant", action="store_true")

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
    

    ag_gemm = parser.add_argument_group('ag-gemm')
    ag_gemm.add_argument(
        "--fast_accum", default=False, action="store_true", help="fp8 use fast accum"
    )
    ag_gemm.add_argument("--sm_margin", default=0, type=int, help="sm margin")

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

EP_GROUP = None
EP_GROUPS = None

def init_ep_group(tp_group, rank, world_size: int, ep_size: int):
    assert world_size % ep_size == 0, f"{world_size} % {ep_size} != 0"
    global EP_GROUP, EP_GROUPS
    assert EP_GROUP is None, "EP_GROUP already initialized"

    assert tp_group.world_size % ep_size == 0, f"{tp_group.world_size} % {ep_size} != 0"
    ffn_tp_size = tp_group.world_size // ep_size

    temp_groups = []
    for i in range(ffn_tp_size):
        ranks = list(range(i, world_size, ffn_tp_size))
        temp_groups.append(ranks)

    ep_groups = []
    for group in temp_groups:
        for i in range(0, len(group), ep_size):
            ep_groups.append(group[i : i + ep_size])

    for ranks in ep_groups:
        group = torch.distributed.new_group(ranks, backend='nccl')
        if rank in ranks:
            EP_GROUP = group
    EP_GROUPS = ep_groups


@torch.compile(dynamic=True, backend=current_platform.simple_compile_backend)
def token_choice_with_bias(hidden_states: torch.Tensor,
                 gating_output: torch.Tensor,
                 topk: int,
                 renormalize: bool):

    assert hidden_states.shape[0] == gating_output.shape[0], (
        "Number of tokens mismatch")

    scores = gating_output.sigmoid()
    topk_weights, topk_ids = torch.topk(scores, k=topk, dim=-1, sorted=False)

    if renormalize:
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

    return topk_weights.to(torch.float32), topk_ids.to(torch.int32)

# def init_worker_distributed_environment(
#     parallel_config,
#     rank: int,
#     distributed_init_method: Optional[str] = None,
#     local_rank: int = -1,
# ) -> None:
#     """Initialize the distributed environment."""
#     set_custom_all_reduce(not parallel_config.disable_custom_all_reduce)
# 
#     init_distributed_environment(parallel_config.world_size, rank,)
#     # init_distributed_environment(parallel_config.world_size, rank,#  distributed_init_method, local_rank)
# 
#     ensure_model_parallel_initialized(parallel_config.tensor_parallel_size,
#                                       parallel_config.pipeline_parallel_size)

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

def gen(seq_lens, num_heads, num_blocks, block_size, head_size,
        sliding_window, dtype, q_dtype, device, tp_size):
    num_seqs = len(seq_lens)
    query_lens = [x[0] for x in seq_lens]
    kv_lens_list = [x[1] for x in seq_lens]

    num_query_heads = num_heads[0]
    num_kv_heads = num_heads[1]
    assert num_query_heads % num_kv_heads == 0
    assert num_kv_heads % tp_size == 0
    assert num_query_heads % tp_size == 0

    # NOTE: QKV is Column-parallel, so num_head is divided by tp_size
    num_query_heads = num_query_heads // tp_size
    num_kv_heads = num_kv_heads // tp_size

    max_query_len = max(query_lens)
    max_kv_len = max(kv_lens_list)
    window_size = ((sliding_window - 1, 0) if sliding_window is not None else
                   (-1, -1))
    scale = head_size**-0.5

    query = torch.randn(sum(query_lens),
                        num_query_heads,
                        head_size,
                        dtype=dtype,
                        device=device)
    key_cache = torch.randn(num_blocks,
                            block_size,
                            num_kv_heads,
                            head_size,
                            dtype=dtype,
                            device=device)
    value_cache = torch.randn_like(key_cache, device=device)

    cu_query_lens = torch.tensor([0] + query_lens,
                                 dtype=torch.int32,
                                 device=device).cumsum(dim=0,
                                                       dtype=torch.int32)
    kv_lens = torch.tensor(kv_lens_list, dtype=torch.int32, device=device)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(0,
                                 num_blocks,
                                 (num_seqs, max_num_blocks_per_seq),
                                 dtype=torch.int32,
                                 device=device)

    out = torch.empty_like(query, device=device)

    maybe_quantized_query = query
    maybe_quantized_key_cache = key_cache
    maybe_quantized_value_cache = value_cache
    q_descale = None
    k_descale = None
    v_descale = None
    if q_dtype is not None:
        maybe_quantized_query = query.to(q_dtype)
        maybe_quantized_key_cache = key_cache.to(q_dtype)
        maybe_quantized_value_cache = value_cache.to(q_dtype)

        scale_shape = (num_seqs, num_kv_heads)
        q_descale = torch.ones(scale_shape, dtype=torch.float32, device=device)
        k_descale = torch.ones(scale_shape, dtype=torch.float32, device=device)
        v_descale = torch.ones(scale_shape, dtype=torch.float32, device=device)

    soft_cap = 0
    fa_version = 3
    return (maybe_quantized_query, maybe_quantized_key_cache, maybe_quantized_value_cache,
            out, cu_query_lens, kv_lens, max_query_len, max_kv_len, scale, window_size,
            block_tables, soft_cap, fa_version, q_descale, k_descale, v_descale)

def init_moe_weight(moe_layer):
    with torch.no_grad():
        # Initialize weights. Note: float8 types don't have normal_ but we can
        # create in float32 and cast.
        moe_layer.w13_weight.copy_(
            torch.randn_like(moe_layer.w13_weight, dtype=torch.float32).to(moe_layer.w13_weight.dtype),
        )
        moe_layer.w2_weight.copy_(
            torch.randn_like(moe_layer.w2_weight, dtype=torch.float32).to(moe_layer.w2_weight.dtype),
        )

        # Initialize scales with random values around 1.0
        # These checks ensure we only initialize parameters that were created.
        if hasattr(moe_layer, "w13_weight_scale") and moe_layer.w13_weight_scale is not None:
            moe_layer.w13_weight_scale.uniform_(0.9, 1.1)

        if hasattr(moe_layer, "w2_weight_scale") and moe_layer.w2_weight_scale is not None:
            moe_layer.w2_weight_scale.uniform_(0.9, 1.1)

        # Initialize input scales if they exist
        if hasattr(moe_layer.quant_method, "static_input_scales"):
            if moe_layer.quant_method.static_input_scales:
                if hasattr(moe_layer, "w13_input_scale") and moe_layer.w13_input_scale is not None:
                    moe_layer.w13_input_scale.uniform_(0.9, 1.1)
                
                if hasattr(moe_layer, "w2_input_scale") and moe_layer.w2_input_scale is not None:
                    moe_layer.w2_input_scale.uniform_(0.9, 1.1)

def init_linear_weight(linear_layer):
    with torch.no_grad():
        # Initialize weights. Note: float8 types don't have normal_ but we can
        # create in float32 and cast.
        linear_layer.weight.copy_(
            torch.randn_like(linear_layer.weight, dtype=torch.float32).to(linear_layer.weight.dtype),
        )

        # Initialize scales with random values around 1.0
        # These checks ensure we only initialize parameters that were created.
        if hasattr(linear_layer, "input_scale") and linear_layer.input_scale is not None:
            linear_layer.input_scale.uniform_(0.9, 1.1)
        else:
            linear_layer.input_scale = None

        if hasattr(linear_layer, "weight_scale") and linear_layer.weight_scale is not None:
            linear_layer.weight_scale.uniform_(0.9, 1.1)

        if hasattr(linear_layer, "scheme"):
            # out dtype should be bfloat16 even for fp8, FIXME fp8 linear has some problem
            linear_layer.scheme.out_dtype = torch.bfloat16


def test_allclose(tensor1, tensor2, name, threshold_map=None):
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
        raise ValueError(f"{name} Tensor dtypes don't match: {tensor1.dtype} vs {tensor2.dtype}")
    
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
            ref_out, test_out, name,
        )

def test_results(results):
    for output_name, result in results.items():
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

@torch.no_grad()
def vllm_forward(args,
                o_proj, vllm_moe_layer,
                hidden_size_per_tp, router_logits,
                q, k, v, out, 
                ):
    attn_out = out.view(-1, hidden_size_per_tp)
    # print_rank0(f'q: {q.shape}, k: {k.shape}, v: {v.shape}, out: {out.shape}, attn_out: {attn_out.shape}')
    # print_rank0(f'parllel? {o_proj.input_is_parallel}, in {o_proj.input_size_per_partition}, out {o_proj.output_size_per_partition}, reduce {o_proj.reduce_results}')

    proj_out, _ = o_proj(attn_out)  # applied all-reduce
    # print_rank0(f'proj_out: {proj_out.shape}, router_logits: {router_logits.shape}')
    # print_rank0(f'{o_proj.weight}, {attn_out}, {proj_out}')

    vllm_out = vllm_moe_layer(proj_out.clone(), router_logits) # NOTE: inplace
    # print_rank0(f'vllm_out: {vllm_out.shape}')
    return None, proj_out, vllm_out

@torch.no_grad()
def breakdown_forward(args,
                o_proj, vllm_moe_layer,
                hidden_size_per_tp, router_logits,
                q, k, v, out, 
            ):
    attn_out = out.view(-1, hidden_size_per_tp)
    # print_rank0(f'q: {q.shape}, k: {k.shape}, v: {v.shape}, out: {out.shape}, attn_out: {attn_out.shape}')
    # print_rank0(f'parllel? {o_proj.input_is_parallel}, in {o_proj.input_size_per_partition}, out {o_proj.output_size_per_partition}, reduce {o_proj.reduce_results}')

    assert o_proj.bias is None
    partial_out = torch.nn.functional.linear(attn_out, o_proj.weight)

    ## all-reduce
    # rs_out = None
    # proj_out = tensor_model_parallel_all_reduce(partial_out)
    # print_rank0(f'partial_out: {partial_out.shape}, proj_out: {proj_out.shape}, reduce: {o_proj.reduce_results}, tp_size: {o_proj.tp_size}, weights: {o_proj.weight.shape}, bias: {o_proj.bias}')

    ## RS+AG
    rs_out = tensor_model_parallel_reduce_scatter(partial_out,0)
    proj_out = tensor_model_parallel_all_gather(rs_out,0)
    # print_rank0(f'partial_out: {partial_out.shape}, rs_out: {rs_out.shape}, proj_out: {proj_out.shape},')

    ## RS+AG (high precision)
    # rs_out = tensor_model_parallel_reduce_scatter(partial_out.float(),0)
    # proj_out_fp32 = tensor_model_parallel_all_gather(rs_out,0)
    # proj_out = proj_out_fp32.bfloat16()
    # print_rank0(f'partial_out: {partial_out.shape}, rs_out: {rs_out.shape}, proj_out: {proj_out.shape},')

    ## RS+AG (torch)
    # tp_group = get_tp_group().device_group
    # tp_size = get_tensor_model_parallel_world_size()
    # rs_out_shape = list(partial_out.shape)
    # rs_out_shape[0] //= tp_size
    # rs_out = torch.empty(rs_out_shape, dtype=partial_out.dtype, device=partial_out.device)
    # dist.reduce_scatter_tensor(rs_out, partial_out, group=tp_group)
    # proj_out = torch.empty_like(partial_out)
    # dist.all_gather_into_tensor(proj_out, rs_out, group=tp_group)
    # print_rank0(f'partial_out: {partial_out.shape}, rs_out: {rs_out.shape}, proj_out: {proj_out.shape}')

    vllm_out = vllm_moe_layer(proj_out.clone(), router_logits)
    # print_rank0(f'vllm_out: {vllm_out.shape}')
    return rs_out, proj_out, vllm_out


def _bf16_moe(
    hidden_states: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    activation: str = "silu",
    global_num_experts: int = -1,
):
    from vllm.model_executor.layers.fused_moe.fused_moe import invoke_fused_moe_kernel
    assert hidden_states.dtype == torch.bfloat16

    num_tokens = hidden_states.shape[0]
    E, N, _ = w1.shape
    K = w2.shape[1]
    if global_num_experts == -1:
        global_num_experts = E
    top_k_num = topk_ids.shape[1]
    # We execute the fused_moe kernel in chunks to circumvent this issue:
    # https://github.com/vllm-project/vllm/issues/5938
    CHUNK_SIZE = 65536
    M = min(num_tokens, CHUNK_SIZE)
    config = {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}
    # print_rank0(f'w1: {w1.shape}, w2: {w2.shape}, {M}, {top_k_num}, {N}, {K}')

    # do not REUSE
    intermediate_cache1 = torch.empty((M, top_k_num, N),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache2 = torch.empty((M * top_k_num, N // 2),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    intermediate_cache3 = torch.empty((M, top_k_num, K),
                                      device=hidden_states.device,
                                      dtype=hidden_states.dtype)
    apply_router_weight_on_input = False
    compute_type = tl.bfloat16
    out_hidden_states = torch.empty_like(hidden_states)

    for chunk in range((num_tokens // CHUNK_SIZE) + 1):
        begin_chunk_idx, end_chunk_idx = (chunk * CHUNK_SIZE,
                                          min((chunk + 1) * CHUNK_SIZE,
                                              num_tokens))
        curr_hidden_states = hidden_states[begin_chunk_idx:end_chunk_idx]
        tokens_in_chunk, _ = curr_hidden_states.shape

        if tokens_in_chunk == 0:
            break

        if tokens_in_chunk < CHUNK_SIZE and chunk > 0:
            # Adjust the intermediate cache size and config for the last
            # chunk. Note that in most cases we only have one chunk
            # so the cache size and config are already set correctly and
            # do not need to be adjusted.
            intermediate_cache1 = intermediate_cache1[:tokens_in_chunk]
            intermediate_cache2 = intermediate_cache2[:tokens_in_chunk *
                                                      topk_ids.shape[1]]
            intermediate_cache3 = intermediate_cache3[:tokens_in_chunk]
            # config = get_config_func(tokens_in_chunk)

        curr_topk_ids = topk_ids[begin_chunk_idx:end_chunk_idx]
        curr_topk_weights = topk_weights[begin_chunk_idx:end_chunk_idx]

        qcurr_hidden_states, a1q_scale = curr_hidden_states, None

        sorted_token_ids, expert_ids, num_tokens_post_padded = (
            moe_align_block_size(curr_topk_ids, config['BLOCK_SIZE_M'],
                                 global_num_experts, expert_map=None))

        invoke_fused_moe_kernel(qcurr_hidden_states,
                                w1,
                                intermediate_cache1,
                                a1q_scale,
                                None,
                                None,
                                curr_topk_weights,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                apply_router_weight_on_input,
                                top_k_num,
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=False,
                                use_int8_w8a8=False,
                                use_int8_w8a16=False,
                                use_int4_w4a16=False,
                                per_channel_quant=False,
                                block_shape=None)

        if activation == "silu":
            torch.ops._C.silu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        elif activation == "gelu":
            torch.ops._C.gelu_and_mul(intermediate_cache2,
                                      intermediate_cache1.view(-1, N))
        else:
            raise ValueError(f"Unsupported FusedMoe activation: {activation}")

        qintermediate_cache2, a2q_scale = intermediate_cache2, None
        invoke_fused_moe_kernel(qintermediate_cache2,
                                w2,
                                intermediate_cache3,
                                a2q_scale,
                                None,
                                None,
                                curr_topk_weights,
                                sorted_token_ids,
                                expert_ids,
                                num_tokens_post_padded,
                                not apply_router_weight_on_input,
                                1,
                                config,
                                compute_type=compute_type,
                                use_fp8_w8a8=False,
                                use_int8_w8a8=False,
                                use_int8_w8a16=False,
                                use_int4_w4a16=False,
                                per_channel_quant=False,
                                block_shape=None)

        ops.moe_sum(intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states[begin_chunk_idx:end_chunk_idx])
        
        ## XXX accumulate outside the kernel
        ## the weighted match - but sum does not match...
        # cache3 = torch.empty_like(intermediate_cache3)
        # invoke_fused_moe_kernel(qintermediate_cache2,
        #                         w2,
        #                         cache3,
        #                         a2q_scale,
        #                         None,
        #                         None,
        #                         curr_topk_weights,
        #                         sorted_token_ids,
        #                         expert_ids,
        #                         num_tokens_post_padded,
        #                         apply_router_weight_on_input,
        #                         1,
        #                         config,
        #                         compute_type=compute_type,
        #                         use_fp8_w8a8=False,
        #                         use_int8_w8a8=False,
        #                         use_int8_w8a16=False,
        #                         use_int4_w4a16=False,
        #                         per_channel_quant=False,
        #                         block_shape=None)
        # # weighted_cache3 = torch.zeros((M, 8, K), device=cache3.device, dtype=cache3.dtype)
        # # for i in range(M):
        # #     for j in range(8): # topk
        # #         weighted_cache3[i][j] = cache3[i][j] * topk_weights[i][j]
        # weighted_cache3 = cache3 * topk_weights.view(M, -1, 1).to(cache3.dtype)
        
        # # out = weighted_cache3.sum(1)
        # out = torch.empty((M, K), device=out_hidden_states.device, dtype=out_hidden_states.dtype)
        # ops.moe_sum(weighted_cache3.view(*weighted_cache3.shape),
        #             out)
        # # print_rank0(f'topk_weights={topk_weights.shape}, weighted_cache3={weighted_cache3.shape}, out={out.shape}, intermediate_cache3={intermediate_cache3.shape}, out_hidden_states={out_hidden_states.shape}')
        # r={}
        # test_tensors(weighted_cache3, intermediate_cache3, 'cache3', r)
        # test_tensors(out, out_hidden_states, 'out', r)
        # test_results(r)
        # raise

    assert chunk == 0, 'we only want 1 step now'
    return intermediate_cache1, intermediate_cache2, intermediate_cache3, out_hidden_states

def _fp8_moe():
    return 

@torch.no_grad()
def gemm_rs_moe_forward(args,
                o_proj, vllm_moe_layer: FusedMoE,
                hidden_size_per_tp, router_logits,
                q, k, v, out, 
            ):
    attn_out = out.view(-1, hidden_size_per_tp)

    # reduce-scatter + GEMM
    reduce_scatter_option = prepare_rs(args)
    tp_process_group = get_tp_group().device_group
    M = attn_out.size(0) # in 
    N = args.hidden_size # out
    is_fp8 = flux.util.is_fp8_dtype(attn_out.dtype)
    is_s8_dequant = attn_out.dtype == torch.int8
    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else attn_out.dtype
    # print(f'rank {dist.get_rank()} attn_out: {attn_out.device}, o_proj.weight: {o_proj.weight.device}, {o_proj.weight.shape}, {M}, {output_dtype}')
    dist.barrier()

    gemm_rs_op = flux.GemmRS(
        tp_process_group,
        1,  # NNODE
        (M + 1024 - 1) // 1024 * 1024,
        N,
        attn_out.dtype,
        output_dtype,
        transpose_weight=args.transpose_weight,
        fuse_reduction=args.fuse_reduction,
        ring_reduction=args.ring_reduction,
    )
    rs_out = gemm_rs_op.forward(
        attn_out,
        o_proj.weight,
        bias=None,
        input_scale=None,
        weight_scale=None,
        output_scale=None,
        fast_accum=False,
        reduce_scatter_option=reduce_scatter_option,
    )

    proj_out = tensor_model_parallel_all_gather(rs_out,0)
    # print_rank0(rs_out.shape, clone.shape, router_logits.shape)
    topk_weights, topk_ids, token_expert_indices = fused_topk(
        proj_out,
        gating_output=router_logits,
        topk=vllm_moe_layer.top_k,
        renormalize=vllm_moe_layer.renormalize,
        indices_type=torch.uint32,
    )
    # print_rank0(topk_weights.shape)
    # print_rank0(topk_ids.shape)
    # print_rank0(topk_weights)
    # print_rank0(topk_ids)
    # print_rank0(topk_ids.int().max())
    # choosed_experts, scatter_index, splits_gpu = prepare_scatter_index_split_gpu(topk_ids, args.num_experts)
    # print_rank0(scatter_index.shape)
    # print_rank0(scatter_index)
    # print_rank0(splits_gpu.shape)
    # print_rank0(splits_gpu.sum())

    dtype = vllm_moe_layer.w13_weight.dtype
    if dtype == torch.bfloat16:
        c1, c2, c3, partial_vllm_out = _bf16_moe(proj_out, 
            vllm_moe_layer.w13_weight,
            vllm_moe_layer.w2_weight,
            topk_weights,
            topk_ids,
            activation='silu',  # vllm_moe_layer.activation
            global_num_experts=vllm_moe_layer.global_num_experts,
        )
    elif dtype == torch.float8_e4m3fn:
        # TODO breakdown
        partial_vllm_out = vllm_moe_layer.quant_method.fused_experts(
                clone,
                vllm_moe_layer.w13_weight,
                vllm_moe_layer.w2_weight,
                topk_weights,
                topk_ids,
                activation=vllm_moe_layer.activation,
                global_num_experts=vllm_moe_layer.global_num_experts,
                expert_map=vllm_moe_layer.expert_map,
                w1_scale=vllm_moe_layer.w13_weight_scale,
                w2_scale=vllm_moe_layer.w2_weight_scale,
                a1_scale=vllm_moe_layer.w13_input_scale,
                a2_scale=vllm_moe_layer.w2_input_scale,
        )
    else:
        raise   
    # RS+AG
    rs_MoE = tensor_model_parallel_reduce_scatter(partial_vllm_out,0)
    vllm_out = tensor_model_parallel_all_gather(rs_MoE,0)
    # AR
    # vllm_out = tensor_model_parallel_all_reduce(partial_vllm_out)
    print_rank0(f'MOE - w13: {vllm_moe_layer.w13_weight.shape}, w2: {vllm_moe_layer.w2_weight.shape}')
    print_rank0(f'MOE - inter_cache1: {c1.shape}, inter_cache2: {c2.shape}, inter_cache3: {c3.shape}, vllm_out: {vllm_out.shape}')
    return rs_out, proj_out, vllm_out, topk_weights, topk_ids.int(), c1, c2, c3, partial_vllm_out, rs_MoE


def prepare_rs(args):
    reduce_scatter_option = ReduceScatterOption()
    reduce_scatter_option.use_1d_ring = args.use_1d_ring
    reduce_scatter_option.use_p2p_read = args.use_p2p_read
    reduce_scatter_option.use_cudaMemcpyAsync = args.use_cudaMemcpyAsync
    reduce_scatter_option.use_gemmk = args.use_gemmk
    reduce_scatter_option.per_tile_flags = args.per_tile_flags
    reduce_scatter_option.num_blocks = args.reduce_scatter_blocks
    reduce_scatter_option.ring_mode = {
        "ring1d": flux.RingMode.Ring1D,
        "ring2d": flux.RingMode.Ring2D,
    }.get(args.ring_mode, None)
    return reduce_scatter_option

def prepare_scatter_index_split_gpu(topk_ids, num_experts):
    # shape: [M, topk]
    choosed_experts = topk_ids.int()
    scatter_index = choosed_experts.flatten().argsort(stable=True).argsort().int().view(choosed_experts.shape)
    splits_gpu = torch.bincount(choosed_experts.view(-1), minlength=num_experts).to(
        torch.int32
    )  # this step is synchronized
    return choosed_experts, scatter_index, splits_gpu


@torch.no_grad()
def gemmRS_AGmoe_forward(args,
                o_proj, vllm_moe_layer: FusedMoE,
                hidden_size_per_tp, router_logits,
                q, k, v, out, 
            ):
    attn_out = out.view(-1, hidden_size_per_tp)

    # PREP
    M = attn_out.size(0) # in 
    hidden_size = args.hidden_size # out
    E, N, _K = vllm_moe_layer.w13_weight.shape
    K = vllm_moe_layer.w2_weight.shape[1]
    assert K == _K
    tp_process_group = get_tp_group().device_group

    intermediate_cache1 = torch.zeros(M, args.topk, N, dtype=attn_out.dtype, device=attn_out.device)
    intermediate_cache2 = torch.empty((M * args.topk, N // 2), device=attn_out.device, dtype=attn_out.dtype)
    intermediate_cache3 = torch.empty((M, args.topk, K), device=attn_out.device, dtype=attn_out.dtype)

    is_fp8 = flux.util.is_fp8_dtype(attn_out.dtype)
    is_s8_dequant = attn_out.dtype == torch.int8
    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else attn_out.dtype
    # print(f'rank {dist.get_rank()} attn_out: {attn_out.device}, o_proj.weight: {o_proj.weight.device}, {o_proj.weight.shape}, {M}, {output_dtype}')

    # OUT_PROJECT
    reduce_scatter_option = prepare_rs(args)
    gemm_rs_op = flux.GemmRS(
        tp_process_group,
        1,  # NNODE
        (M + 1024 - 1) // 1024 * 1024,
        hidden_size,
        attn_out.dtype,
        output_dtype,
        transpose_weight=args.transpose_weight,
        fuse_reduction=args.fuse_reduction,
        ring_reduction=args.ring_reduction,
    )
    rs_out = gemm_rs_op.forward(
        attn_out,
        o_proj.weight,
        bias=None,
        input_scale=None,
        weight_scale=None,
        output_scale=None,
        fast_accum=False,
        reduce_scatter_option=reduce_scatter_option,
    )

    # ROUTER
    # TODO this can be skip; see the fused_topk, but use for convenience
    proj_out = tensor_model_parallel_all_gather(rs_out,0)
    # print_rank0(rs_out.shape,proj_out.shape, router_logits.shape)
    topk_weights, topk_ids, token_expert_indices = fused_topk(
        proj_out,
        gating_output=router_logits,
        topk=vllm_moe_layer.top_k,
        renormalize=vllm_moe_layer.renormalize,
        indices_type=torch.uint32,
    )
    choosed_experts, scatter_index, splits_gpu = prepare_scatter_index_split_gpu(topk_ids, E)

    # MOE0
    input_dtype = rs_out.dtype
    output_dtype = torch.bfloat16 if is_fp8 else input_dtype
    moe_args = flux.MoeArguments(
        max_ntokens=args.batch_size * args.seq_len,
        hidden=hidden_size,
        ffn_hidden=args.intermediate_size * 2,  # FuseMoE's layer 0 doubles this, then act reduce by half
        nexperts=E,
        topk=args.topk,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )
    extra_args = {}
    global EP_GROUP
    tp_env = flux.DistEnvTPWithEP(tp_group=tp_process_group, nnodes=1, ep_group=EP_GROUP)
    ag_moe_op = flux.GemmGroupedV3AGScatter(tp_env=tp_env, moe_args=moe_args)
    ag_moe_op.clear_buffers()
    ag_out = torch.empty_like(proj_out)
    c1 = ag_moe_op.forward(
        inputs_shard=rs_out,
        weights=vllm_moe_layer.w13_weight,
        splits_gpu=splits_gpu,
        scatter_index=scatter_index,
        fast_accum=args.fast_accum,
        sm_margin=args.sm_margin,
        output_scale=None,
        outputs_buf=None,
        allgather_output=ag_out,
        **extra_args,
    )
    # unpermute c1 into intermediate_cache1 for verification; c1: [M * topk, N]
    for i in range(M):
        indices = scatter_index[i]
        intermediate_cache1[i] = c1[indices]
    # intermediate2
    torch.ops._C.silu_and_mul(intermediate_cache2,
                                intermediate_cache1.view(-1, N))
    # intermediate3 - token-centric
    tmp_cache2 = intermediate_cache2.reshape(M, args.topk, N//2)
    for i in range(M):
        expert_ids = topk_ids[i]
        for j in range(args.topk):
            expert_weight = vllm_moe_layer.w2_weight[expert_ids[j]]
            out = expert_weight @ tmp_cache2[i][j]
            intermediate_cache3[i][j] = out

    # ACCU expert results

    ## just sum
    # FIXME: for topk=2,3,4, the vllm_op moe_sum != torch.sum ????? 
    # vllm_partial_out = torch.empty((M, K), device=attn_out.device, dtype=attn_out.dtype)
    # ops.moe_sum(intermediate_cache3,
    #             vllm_partial_out)
    # vllm_partial_out = intermediate_cache3.sum(1)

    ## apply_router_weight_on_input, XXX triton's kernel seems to lose some precision, see vllm's test_moe
    intermediate_cache3 = intermediate_cache3 * topk_weights.view(M, -1, 1).to(intermediate_cache3.dtype)
    vllm_partial_out = intermediate_cache3.sum(1)
    print_rank0(f'intermediate_cache1: {intermediate_cache1.shape}, intermediate_cache2: {intermediate_cache2.shape}, intermediate_cache3: {intermediate_cache3.shape}, vllm_partial_out: {vllm_partial_out.shape}')

    # AR
    # token_centric_out = tensor_model_parallel_all_reduce(vllm_partial_out)

    # RS+AG
    # token_centric_out_rs = tensor_model_parallel_reduce_scatter(vllm_partial_out.float(),0)
    # token_centric_out_rs = token_centric_out_rs.bfloat16()
    token_centric_out_rs = tensor_model_parallel_reduce_scatter(vllm_partial_out,0)
    token_centric_out = tensor_model_parallel_all_gather(token_centric_out_rs,0)

    return rs_out, ag_out, token_centric_out, topk_weights, topk_ids.int(), \
        intermediate_cache1, intermediate_cache2, intermediate_cache3, vllm_partial_out, token_centric_out_rs



@torch.no_grad()
def flux_forward(args,
                o_proj, vllm_moe_layer: FusedMoE,
                hidden_size_per_tp, router_logits,
                q, k, v, out, 
            ):
    attn_out = out.view(-1, hidden_size_per_tp)

    # PREP
    M = attn_out.size(0) # in 
    hidden_size = args.hidden_size # out
    E, N, _K = vllm_moe_layer.w13_weight.shape
    K = vllm_moe_layer.w2_weight.shape[1]
    assert K == _K
    tp_process_group = get_tp_group().device_group
    is_fp8 = flux.util.is_fp8_dtype(attn_out.dtype)
    is_s8_dequant = attn_out.dtype == torch.int8
    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else attn_out.dtype
    # print(f'rank {dist.get_rank()} attn_out: {attn_out.device}, o_proj.weight: {o_proj.weight.device}, {o_proj.weight.shape}, {M}, {output_dtype}')

    # OUT_PROJECT
    reduce_scatter_option = prepare_rs(args)
    gemm_rs_op = flux.GemmRS(
        tp_process_group,
        1,  # NNODE
        (M + 1024 - 1) // 1024 * 1024,
        hidden_size,
        attn_out.dtype,
        output_dtype,
        transpose_weight=args.transpose_weight,
        fuse_reduction=args.fuse_reduction,
        ring_reduction=args.ring_reduction,
    )
    rs_out = gemm_rs_op.forward(
        attn_out,
        o_proj.weight,
        bias=None,
        input_scale=None,
        weight_scale=None,
        output_scale=None,
        fast_accum=False,
        reduce_scatter_option=reduce_scatter_option,
    )

    # ROUTER
    # TODO this can be skip; see the fused_topk, but use for convenience
    proj_out = tensor_model_parallel_all_gather(rs_out,0)
    # print_rank0(rs_out.shape,proj_out.shape, router_logits.shape)
    topk_weights, topk_ids, token_expert_indices = fused_topk(
        proj_out,
        gating_output=router_logits,
        topk=vllm_moe_layer.top_k,
        renormalize=vllm_moe_layer.renormalize,
        indices_type=torch.uint32,
    )
    choosed_experts, scatter_index, splits_gpu = prepare_scatter_index_split_gpu(topk_ids, E)

    # MOE0
    input_dtype = rs_out.dtype
    output_dtype = torch.bfloat16 if is_fp8 else input_dtype
    moe_args = flux.MoeArguments(
        max_ntokens=args.batch_size * args.seq_len,
        hidden=hidden_size,
        ffn_hidden=args.intermediate_size * 2,  # FuseMoE's layer 0 doubles this, then act reduce by half
        nexperts=E,
        topk=args.topk,
        input_dtype=input_dtype,
        output_dtype=output_dtype,
    )
    extra_args = {}
    global EP_GROUP
    tp_env = flux.DistEnvTPWithEP(tp_group=tp_process_group, nnodes=1, ep_group=EP_GROUP)
    ag_moe_op = flux.GemmGroupedV3AGScatter(tp_env=tp_env, moe_args=moe_args)
    ag_moe_op.clear_buffers()
    ag_out = torch.empty_like(proj_out)
    c1 = ag_moe_op.forward(
        inputs_shard=rs_out,
        weights=vllm_moe_layer.w13_weight,
        splits_gpu=splits_gpu,
        scatter_index=scatter_index,
        fast_accum=args.fast_accum,
        sm_margin=args.sm_margin,
        output_scale=None,
        outputs_buf=None,
        allgather_output=ag_out,
        **extra_args,
    )

    c2 = torch.empty(size=(M*args.topk, N // 2), dtype=c1.dtype, device=c1.device)
    torch.ops._C.silu_and_mul(c2, c1)

    # MOE1
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    routing_idx = scatter_index.view(-1)
    flux_rs_op = flux.GemmGroupedV3GatherRS(E, 
                                            M*args.topk, 
                                            hidden_size, 
                                            args.topk, 
                                            rank,
                                            world_size, 
                                            args.tp_size, 
                                            args.ep_size, 
                                            1,
                                        )
    print_rank0(f'[Flux-MoE] c2: {c2.shape}, w2: {vllm_moe_layer.w2_weight.shape}, routing_idx: {routing_idx.shape}')
    c3 = flux_rs_op.forward_gather_rs(
        input=c2,
        weight=vllm_moe_layer.w2_weight,
        splits_cpu=splits_gpu.cpu(),
        routing_idx=routing_idx,
    )
    # final - overlap with next layer
    vllm_out = tensor_model_parallel_all_gather(c3,0)
    print_rank0(f'[Flux-MoE] c3: {c3.shape}, vllm_out: {vllm_out.shape}')

    return rs_out, ag_out, vllm_out, topk_weights, topk_ids.int(), \
        None, None, None, None, c3


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
    dist.barrier()


    ## vllm is either TP or EP?
    tp = get_tp_group()
    print_rank0(f'tp: {tp.rank_in_group}, {tp.ranks}')
    ep = get_ep_group()
    print_rank0(f'ep: {ep.rank_in_group}, {ep.ranks}')
    pp = get_pp_group()
    print_rank0(f'pp: {pp.rank_in_group}, {pp.ranks}')

    ## we manage EP explicitly since vLLM is either TP or EP
    init_ep_group(get_tp_group(), rank, world_size, args.ep_size)
    print_rank0(f'EP GROUPS: ', EP_GROUPS)

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
    M = args.batch_size * args.seq_len
    router_logits = torch.randn(M, args.num_experts, device=device, dtype=dtype)
    seq_lens = [(args.seq_len, args.seq_len) for _ in range(args.batch_size)]
    num_heads = (32, 32)
    hidden_size_per_tp = args.hidden_size // args.tp_size
    assert args.hidden_size % args.tp_size == 0 
    q, k, v, \
        out, cu_query_lens, kv_lens, max_query_len, max_kv_len, scale, window_size, block_tables, soft_cap, fa_version, \
        q_descale, k_descale, v_descale = \
            gen(seq_lens,
                num_heads,
                32768,
                16,
                128,
                None,
                dtype=dtype,  # q,k,v dtype
                q_dtype=None,  # args.q_dtype,
                device=device,
                tp_size=args.tp_size,
            )
    # torch.distributed.breakpoint(0)
    dist.barrier()

    # model
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import CompressedTensorsConfig
    from compressed_tensors.quantization import QuantizationArgs  
    ## make a fake config that falls into the quant_method
    target_scheme = {'Linear': {'weights': QuantizationArgs(num_bits=8, type='float', symmetric=True, group_size=None, strategy='channel', block_structure=None, dynamic=False, actorder=None, observer='minmax', observer_kwargs={}), 
                                'input_activations': QuantizationArgs(num_bits=8, type='float', symmetric=True, group_size=None, strategy='token', block_structure=None, dynamic=True, actorder=None, observer=None, observer_kwargs={})},
                     }
    quant_config = CompressedTensorsConfig(target_scheme_map=target_scheme, 
                                           ignore=['lm_head'], 
                                           quant_format='float-quantized',
                                           sparsity_scheme_map={},
                                           sparsity_ignore_list=[],
                                        )
    vllm_moe_layer = FusedMoE(num_experts=args.num_experts,
                            top_k=args.topk,
                            hidden_size=args.hidden_size,  # sharded by TP
                            intermediate_size=args.intermediate_size,
                            params_dtype=torch.bfloat16,  # input is bf16, but quant will apply to weights
                            reduce_results=True,
                            renormalize=True,
                            #
                            # quant_config=None,  
                            quant_config=quant_config if args.quant else None, # quant default is bf16

                            # tp_size=args.tp_size,
                            # prefix=f"experts",
                            # custom_routing_function=token_choice_with_bias,
                        ).to(device)
    init_moe_weight(vllm_moe_layer)

    o_proj = RowParallelLinear(
        args.hidden_size,  # in: sharded by tp
        args.hidden_size,  # out
        bias=False,
        params_dtype=dtype,

        # TODO fp8 linear has some problem
        # quant_config=quant_config if args.quant else None, 
    ).to(device)
    init_linear_weight(o_proj)
    print_rank0(f'[Linear] {o_proj.weight.shape}, {o_proj.weight.dtype}')
    print_rank0(f'[MoE] {vllm_moe_layer.w13_weight.shape} {vllm_moe_layer.w13_weight.dtype}')
    # print_rank0(f'{o_proj.scheme.out_dtype}, {o_proj.scheme}')
    dist.barrier()

    # forward
    flash_attn_varlen_func(
        q=q, k=k, v=v, out=out,
        cu_seqlens_q=cu_query_lens,
        seqused_k=kv_lens,
        max_seqlen_q=max_query_len,
        max_seqlen_k=max_kv_len,
        softmax_scale=scale,
        causal=True,
        window_size=window_size,
        block_table=block_tables,
        softcap=soft_cap,
        fa_version=fa_version,
        q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
    )
    print_rank0('=='*10 + 'vLLM - breakdown' + '=='*10)
    results = {}
    names = ['rs_out','proj_out', 'moe_out']
    vllm_outs = vllm_forward(args,
                o_proj, vllm_moe_layer,
                hidden_size_per_tp, router_logits,
                q, k, v, out, 
            )
    breakdown_outs = breakdown_forward(args,
                o_proj, vllm_moe_layer,
                hidden_size_per_tp, router_logits,
                q, k, v, out,
            )
    for name, vllm_out, break_out in zip(names, vllm_outs, breakdown_outs):
        test_tensors(vllm_out, break_out, name, results)
        # if name == 'proj_out':
        #     print_rank0(vllm_out)
        #     print_rank0(break_out)
        # print_rank0(vllm_out)
        # print_rank0(break_out)
    test_results(results)
    dist.barrier()

    # gemmRS - breakdown
    print_rank0('=='*10 + 'gemmRS_moe - breakdown' + '=='*10)
    results = {}
    gemmRS_outs = gemm_rs_moe_forward(args,
                 o_proj, vllm_moe_layer,
                 hidden_size_per_tp, router_logits,
                 q, k, v, out, 
                )
    for name, out1, out2 in zip(names, breakdown_outs, gemmRS_outs):
        test_tensors(out1, out2, name, results)
    test_results(results)
    dist.barrier()

    print_rank0('=='*10 + 'gemmRS_moe - gemmRS_AGmoe' + '=='*10)
    results = {}
    gemmRS_AGmoe_outs = gemmRS_AGmoe_forward(args,
                 o_proj, vllm_moe_layer,
                 hidden_size_per_tp, router_logits,
                 q, k, v, out, 
                )
    names = ['rs_out','proj_out', 'moe_out', 'topk_w', 'topk_ids', 'c1', 'c2', 'c3', 'partial_moe_out', 'RS-MOE']
    for name, out1, out2 in zip(names, gemmRS_outs, gemmRS_AGmoe_outs):
        test_tensors(out1, out2, name, results)
    test_results(results)
    dist.barrier()

    print_rank0('=='*10 + 'flux - gemmRS_AGmoe' + '=='*10)
    results = {}
    flux_outs = flux_forward(args,
                o_proj, vllm_moe_layer,
                hidden_size_per_tp, router_logits,
                q, k, v, out, 
                )
    names = ['rs_out','proj_out', 'moe_out', 'topk_w', 'topk_ids', 'c1', 'c2', 'c3', 'partial_moe_out', 'RS-MOE']
    for name, out1, out2 in zip(names, flux_outs, gemmRS_AGmoe_outs):
        test_tensors(out1, out2, name, results)
    test_results(results)
    dist.barrier()

    # bench
    # quantiles = [0.5, 0.2, 0.8]
    # ms, min_ms, max_ms = triton.testing.do_bench(lambda: vllm_moe_layer(hidden_states, router_logits),
    #                                               quantiles=quantiles)

    # print(f'{ms}ms, min: {min_ms}ms, max: {max_ms}ms')

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
