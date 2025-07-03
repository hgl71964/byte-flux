import os
from typing import Optional, Callable, Tuple, List

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

                            init_world_group,
                              )
from vllm.distributed import (ensure_model_parallel_initialized,
                              init_distributed_environment,
                              set_custom_all_reduce)
from vllm.vllm_flash_attn import flash_attn_varlen_func
from vllm.model_executor.layers.linear import RowParallelLinear

# torchrun --nproc_per_node=2 custom_bench/vllm_moe_layer.py
def print_rank0(*args):
    if dist.get_rank() == 0:
        print(*args)

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
    parser.add_argument("--dtype", type=str, default="bfloat16")

    args = parser.parse_args()
    return args

def get_dtype(args):
    if args.dtype == "bfloat16":
        return torch.bfloat16
    elif args.dtype == "float16":
        return torch.float16
    elif args.dtype == "float32":
        return torch.float32
    else:
        raise ValueError(f"Unsupported dtype: {args.dtype}")

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
    dist.init_process_group(backend="nccl")
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


def vllm_forward(o_proj, vllm_moe_layer, 
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
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    attn_out = out.view(-1, hidden_size_per_tp)
    print_rank0(f'q: {q.shape}, k: {k.shape}, v: {v.shape}, out: {out.shape}, attn_out: {attn_out.shape}')
    # print_rank0(f'parllel? {o_proj.input_is_parallel}, in {o_proj.input_size_per_partition}, out {o_proj.output_size_per_partition}, reduce {o_proj.reduce_results}')

    proj_out, _ = o_proj(attn_out)  # NOTE: applied all-reduce
    print_rank0(f'proj_out: {proj_out.shape}, router_logits: {router_logits.shape}')

    vllm_out = vllm_moe_layer(proj_out, router_logits)
    print_rank0(f'vllm_out: {vllm_out.shape}')
    return attn_out, proj_out, vllm_out

def overlap_forward(o_proj, vllm_moe_layer, 
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
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    attn_out = out.view(-1, hidden_size_per_tp)
    print_rank0(f'q: {q.shape}, k: {k.shape}, v: {v.shape}, out: {out.shape}, attn_out: {attn_out.shape}')

    # reduce-scatter + GEMM

    # all-gather + MoE

    return attn_out

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

    # data
    dtype = get_dtype(args)
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
                            custom_routing_function=token_choice_with_bias)
    vllm_moe_layer.to(device)

    o_proj = RowParallelLinear(
        args.hidden_size,  # in NOTE: this will be sharded by tp
        args.hidden_size,  # out
        bias=False,
        params_dtype=dtype,
        quant_config=None, # quant_config=quant_config,
    ).to(device)
    dist.barrier()

    # forward
    vllm_out = vllm_forward(o_proj, vllm_moe_layer,
                 hidden_size_per_tp, router_logits,
                 q, k, v, out, 
                 cu_query_lens, kv_lens, max_query_len, max_kv_len, scale, 
                 window_size, block_tables, soft_cap, fa_version,
                 q_descale, k_descale, v_descale,
                )

    # bench
    # quantiles = [0.5, 0.2, 0.8]
    # ms, min_ms, max_ms = triton.testing.do_bench(lambda: vllm_moe_layer(hidden_states, router_logits),
    #                                               quantiles=quantiles)

    # print(f'{ms}ms, min: {min_ms}ms, max: {max_ms}ms')

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
