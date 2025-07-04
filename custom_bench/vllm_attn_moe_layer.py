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

def determine_expert_map(
        ep_size: int, ep_rank: int,
        global_num_experts: int) -> Tuple[int, Optional[torch.Tensor]]:
    """
        Calculates how many experts should be assigned to each rank for EP and
        creates a mapping from global to local expert index. Experts are
        distributed evenly across ranks. Any remaining are assigned to the
        last rank.

        Args:
            ep_size (int): The size of the expert parallel group
            global_num_experts (int): The total number of experts in the model.

        Returns:
            Tuple[int, Optional[torch.Tensor]]: A tuple containing:
                - local_num_experts (int): The number of experts assigned
                    to the current rank.
                - expert_map (Optional[torch.Tensor]): A tensor of shape
                    (global_num_experts,) mapping from global to local index.
                    Contains -1 for experts not assigned to the current rank.
                    Returns None if ep_size is 1.
        """
    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None)

    local_num_experts = global_num_experts // ep_size

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts, ), -1, dtype=torch.int32)
    # Create a expert map for the local experts
    if ep_rank < (ep_size - 1):
        # Each non-last rank gets local_num_experts experts.
        expert_map[ep_rank * local_num_experts:
                        (ep_rank + 1) * local_num_experts] = \
            torch.arange(0, local_num_experts, dtype=torch.int32)
    else:
        # All remaining experts are assigned to the last rank.
        local_num_experts = (global_num_experts - ep_rank * local_num_experts)

        expert_map[-local_num_experts:] = \
            torch.arange(0, local_num_experts, dtype=torch.int32)
    return (local_num_experts, expert_map)


class FusedMoE_(FusedMoE):

    # @property
    # def ep_size(self):
    #     return self._ep_size

    # @ep_size.setter
    # def ep_size(self, value):
    #     self._ep_size = value

    def __init__(
        self,
        num_experts: int,  # Global number of experts
        top_k: int,
        hidden_size: int,
        intermediate_size: int,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        renormalize: bool = True,
        use_grouped_topk: bool = False,
        num_expert_group: Optional[int] = None,
        topk_group: Optional[int] = None,
        quant_config: Optional[QuantizationConfig] = None,
        tp_size: Optional[int] = None,
        ep_size: Optional[int] = None,
        dp_size: Optional[int] = None,
        prefix: str = "",
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
    ):
        # torch.nn.Module.__init__(self)
        super().__init__(
            num_experts=num_experts,
            top_k=top_k,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            params_dtype=params_dtype,
            reduce_results=reduce_results,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            num_expert_group=num_expert_group,
            topk_group=topk_group,
            quant_config=quant_config,
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
            prefix=prefix,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            activation=activation,
        )
        # self.ep_size=2

    @staticmethod
    def select_experts(hidden_states: torch.Tensor,
                       router_logits: torch.Tensor,
                       top_k: int,
                       use_grouped_topk: bool,
                       renormalize: bool,
                       topk_group: Optional[int] = None,
                       num_expert_group: Optional[int] = None,
                       custom_routing_function: Optional[Callable] = None,
                       scoring_func: str = "softmax",
                       e_score_correction_bias: Optional[torch.Tensor] = None):
        from vllm.model_executor.layers.fused_moe.fused_moe import (
            fused_topk, grouped_topk)

        # DeekSeekv2 uses grouped_top_k
        if use_grouped_topk:
            assert topk_group is not None
            assert num_expert_group is not None
            topk_weights, topk_ids = grouped_topk(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize,
                num_expert_group=num_expert_group,
                topk_group=topk_group,
                scoring_func=scoring_func,
                e_score_correction_bias=e_score_correction_bias)
        elif custom_routing_function is None:
            topk_weights, topk_ids = fused_topk(hidden_states=hidden_states,
                                                gating_output=router_logits,
                                                topk=top_k,
                                                renormalize=renormalize)
        else:
            topk_weights, topk_ids = custom_routing_function(
                hidden_states=hidden_states,
                gating_output=router_logits,
                topk=top_k,
                renormalize=renormalize)

        return topk_weights, topk_ids

    def naive_multicast(self, x: torch.Tensor,
                        cu_tokens_across_dp_cpu: torch.Tensor):
        assert (len(x.shape) == 2)
        buffer = torch.empty((cu_tokens_across_dp_cpu[-1], x.size(1)),
                             device=x.device,
                             dtype=x.dtype)

        start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
            self.dp_rank - 1]
        end = cu_tokens_across_dp_cpu[self.dp_rank]
        buffer[start:end, :].copy_(x)
        for idx in range(get_dp_group().world_size):
            start = 0 if idx == 0 else cu_tokens_across_dp_cpu[idx - 1]
            end = cu_tokens_across_dp_cpu[idx]
            get_dp_group().broadcast(buffer[start:end, :], idx)

        return buffer

    def forward(self, hidden_states: torch.Tensor,
                router_logits: torch.Tensor):
        
        if self.use_direct_call:
            return self.forward_impl(hidden_states, router_logits)
        else:
            return torch.ops.vllm.moe_forward(hidden_states, router_logits,
                                              self.layer_name)

    def forward_impl(self, hidden_states: torch.Tensor,
                     router_logits: torch.Tensor):
        assert self.quant_method is not None

        if self.dp_size > 1:
            cu_tokens_across_dp_cpu = get_forward_context(
            ).dp_metadata.cu_tokens_across_dp_cpu

            hidden_states = self.naive_multicast(hidden_states,
                                                 cu_tokens_across_dp_cpu)
            router_logits = self.naive_multicast(router_logits,
                                                 cu_tokens_across_dp_cpu)

        # Matrix multiply.
        # print(f'forward_impl quant: ', self.quant_method)
        final_hidden_states = self.quant_method.apply(
            layer=self,
            x=hidden_states,
            router_logits=router_logits,
            top_k=self.top_k,
            renormalize=self.renormalize,
            use_grouped_topk=self.use_grouped_topk,
            global_num_experts=self.global_num_experts,
            expert_map=self.expert_map,
            topk_group=self.topk_group,
            num_expert_group=self.num_expert_group,
            custom_routing_function=self.custom_routing_function,
            scoring_func=self.scoring_func,
            e_score_correction_bias=self.e_score_correction_bias,
            activation=self.activation,
        )

        if self.dp_size > 1:
            start = 0 if self.dp_rank == 0 else cu_tokens_across_dp_cpu[
                self.dp_rank - 1]
            end = cu_tokens_across_dp_cpu[self.dp_rank]

            all_hidden_states = get_dp_group().all_reduce(final_hidden_states)
            final_hidden_states = all_hidden_states[start:end, :]

        if self.reduce_results and (self.tp_size > 1 or self.ep_size > 1):
            # Default set to False. (May have to add shared expert outputs.)
            final_hidden_states = tensor_model_parallel_all_reduce(
                final_hidden_states)

        return final_hidden_states

    @classmethod
    def make_expert_params_mapping(
            cls, ckpt_gate_proj_name: str, ckpt_down_proj_name: str,
            ckpt_up_proj_name: str,
            num_experts: int) -> List[Tuple[str, str, int, str]]:

        return [
            # (param_name, weight_name, expert_id, shard_id)
            ("experts.w13_" if weight_name
             in [ckpt_gate_proj_name, ckpt_up_proj_name] else "experts.w2_",
             f"experts.{expert_id}.{weight_name}.", expert_id, shard_id)
            for expert_id in range(num_experts) for shard_id, weight_name in [
                ("w1", ckpt_gate_proj_name),
                ("w2", ckpt_down_proj_name),
                ("w3", ckpt_up_proj_name),
            ]
        ]



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

@torch.no_grad()
def vllm_forward(args,
                o_proj, vllm_moe_layer,
                hidden_size_per_tp, router_logits,
                q, k, v, out, 
                cu_query_lens, kv_lens, max_query_len, max_kv_len, scale, 
                window_size, block_tables, soft_cap, fa_version, 
                q_descale, k_descale, v_descale,
                ):
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
    attn_out = out.view(-1, hidden_size_per_tp)
    print_rank0(f'q: {q.shape}, k: {k.shape}, v: {v.shape}, out: {out.shape}, attn_out: {attn_out.shape}')
    # print_rank0(f'parllel? {o_proj.input_is_parallel}, in {o_proj.input_size_per_partition}, out {o_proj.output_size_per_partition}, reduce {o_proj.reduce_results}')

    proj_out, _ = o_proj(attn_out)  # applied all-reduce
    print_rank0(f'proj_out: {proj_out.shape}, router_logits: {router_logits.shape}')
    # print_rank0(f'{o_proj.weight}, {attn_out}, {proj_out}')

    # TODO vllm_moe_layer will modify in place
    # vllm_out = vllm_moe_layer(proj_out, router_logits)
    # print_rank0(f'vllm_out: {vllm_out.shape}')
    return attn_out, proj_out, None

@torch.no_grad()
def breakdown_forward(args,
                o_proj, vllm_moe_layer,
                hidden_size_per_tp, router_logits,
                q, k, v, out, 
                cu_query_lens, kv_lens, max_query_len, max_kv_len, scale, 
                window_size, block_tables, soft_cap, fa_version, 
                q_descale, k_descale, v_descale,
            ):
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
    attn_out = out.view(-1, hidden_size_per_tp)
    print_rank0(f'q: {q.shape}, k: {k.shape}, v: {v.shape}, out: {out.shape}, attn_out: {attn_out.shape}')
    # print_rank0(f'parllel? {o_proj.input_is_parallel}, in {o_proj.input_size_per_partition}, out {o_proj.output_size_per_partition}, reduce {o_proj.reduce_results}')

    assert o_proj.bias is None

    partial_out = torch.nn.functional.linear(attn_out, o_proj.weight)

    ## all-reduce
    proj_out = tensor_model_parallel_all_reduce(partial_out)
    print_rank0(f'partial_out: {partial_out.shape}, proj_out: {proj_out.shape}, reduce: {o_proj.reduce_results}, tp_size: {o_proj.tp_size}, weights: {o_proj.weight.shape}, bias: {o_proj.bias}')

    ## RS+AG
    reduced_scattered_out = tensor_model_parallel_reduce_scatter(partial_out,0)
    proj_out = tensor_model_parallel_all_gather(reduced_scattered_out,0)
    print_rank0(f'partial_out: {partial_out.shape}, reduced_scattered_out: {reduced_scattered_out.shape}, proj_out: {proj_out.shape},')

    # vllm_out = vllm_moe_layer(proj_out, router_logits)
    # print_rank0(f'vllm_out: {vllm_out.shape}')
    return attn_out, proj_out, None


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

@torch.no_grad()
def flux_forward(args,
                o_proj, vllm_moe_layer,
                hidden_size_per_tp, router_logits,
                q, k, v, out, 
                cu_query_lens, kv_lens, max_query_len, max_kv_len, scale, 
                window_size, block_tables, soft_cap, fa_version, 
                q_descale, k_descale, v_descale,
            ):
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
    attn_out = out.view(-1, hidden_size_per_tp)

    # reduce-scatter + GEMM
    reduce_scatter_option = prepare_rs(args)
    tp_process_group = get_tp_group().device_group
    M = attn_out.size(0) # in 
    N = args.hidden_size # out
    is_fp8 = flux.util.is_fp8_dtype(attn_out.dtype)
    is_s8_dequant = attn_out.dtype == torch.int8
    output_dtype = torch.bfloat16 if is_fp8 or is_s8_dequant else attn_out.dtype

    print(f'rank {dist.get_rank()} attn_out: {attn_out.device}, o_proj.weight: {o_proj.weight.device}, {o_proj.weight.shape}, {M}, {output_dtype}')
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
    output = gemm_rs_op.forward(
        attn_out,
        o_proj.weight,
        bias=None,
        input_scale=None,
        weight_scale=None,
        output_scale=None,
        fast_accum=False,
        reduce_scatter_option=reduce_scatter_option,
    )
    print_rank0(f'gemm RS out: {output.shape}')

    # all-gather 

    # FusedMoE_

    return attn_out, None, None

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
    vllm_moe_layer = FusedMoE_(num_experts=args.num_experts,
                            top_k=args.num_experts_per_tok,
                            hidden_size=args.hidden_size,
                            intermediate_size=args.intermediate_size,
                            params_dtype=dtype,  # TODO fp8?
                            reduce_results=True,
                            renormalize=True,
                            quant_config=None,  # TODO
                            tp_size=args.tp_size,
                            prefix=f"experts",
                            custom_routing_function=token_choice_with_bias).to(device)

    o_proj = RowParallelLinear(
        args.hidden_size,  # in NOTE: this will be sharded by tp
        args.hidden_size,  # out
        bias=False,
        params_dtype=dtype,
        quant_config=None, # quant_config=quant_config,
    ).to(device)
    with torch.no_grad():
        o_proj.weight.data.uniform_(0, 1)
    dist.barrier()


    # forward
    vllm_outs = vllm_forward(args,
                o_proj, vllm_moe_layer,
                hidden_size_per_tp, router_logits,
                q, k, v, out, 
                cu_query_lens, kv_lens, max_query_len, max_kv_len, scale, 
                window_size, block_tables, soft_cap, fa_version,
                q_descale, k_descale, v_descale,
            )
    breakdown_outs = breakdown_forward(args,
                o_proj, vllm_moe_layer,
                hidden_size_per_tp, router_logits,
                q, k, v, torch.empty_like(out), 
                cu_query_lens, kv_lens, max_query_len, max_kv_len, scale, 
                window_size, block_tables, soft_cap, fa_version,
                q_descale, k_descale, v_descale,
            )
    names = ['attn_out', 'proj_out', 'moe_out']
    results = {}
    for name, vllm_out, break_out in zip(names, vllm_outs, breakdown_outs):
        test_tensors(vllm_out, break_out, name, results)

        # if name == 'proj_out':
        #     print_rank0(vllm_out)
        #     print_rank0(break_out)

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

    # flux_outs = flux_forward(args,
    #              o_proj, vllm_moe_layer,
    #              hidden_size_per_tp, router_logits,
    #              q, k, v, out, 
    #              cu_query_lens, kv_lens, max_query_len, max_kv_len, scale, 
    #              window_size, block_tables, soft_cap, fa_version,
    #              q_descale, k_descale, v_descale,
    #             )
    # names = ['attn_out', 'proj_out', 'moe_out']

    # results = {}
    # for name, vllm_out, flux_out in zip(names, vllm_outs, flux_outs):
    #     test_tensors(vllm_out, flux_out, name, results)
    # for output_name, is_close in results.items():
    #     status = "✅ PASS" if is_close else "❌ FAIL"
    #     print(f"  {output_name}: {status}")

    # bench
    # quantiles = [0.5, 0.2, 0.8]
    # ms, min_ms, max_ms = triton.testing.do_bench(lambda: vllm_moe_layer(hidden_states, router_logits),
    #                                               quantiles=quantiles)

    # print(f'{ms}ms, min: {min_ms}ms, max: {max_ms}ms')

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
