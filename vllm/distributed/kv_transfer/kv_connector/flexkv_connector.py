# SPDX-License-Identifier: Apache-2.0
"""
Simple KV Cache Connector for Distributed Machine Learning Inference

The SimpleConnector transfers KV caches between prefill vLLM worker (KV cache
producer) and decode vLLM worker (KV cache consumer) using PyNcclPipe or
MooncakePipe.

But the logic can be extended to support other pipe and lookup buffer.
"""
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

import vllm.envs as envs
from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_lookup_buffer.simple_buffer import (
    SimpleBuffer)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata

logger = init_logger(__name__)


import torch
from flexkv.kvmanager import KVManager
from flexkv.common.config import ModelConfig, CacheConfig
from flexkv.common.debug import debuginfo
import time
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.base import KVConnectorBase
from vllm.distributed.kv_transfer.kv_connector.utils import (
    build_new_prefill_input)
from vllm.logger import init_logger
from vllm.sequence import IntermediateTensors
import os
import nvtx
if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
logger = init_logger(__name__)
nv_cache_debug_level = os.getenv('FLEXKV_LOG_LEVEL', 'INFO')
cpu_blocks = int(os.getenv('FLEXKV_CPU_BLOCKS',1000))
# ssd_blocks = int(os.getenv('NV_CACHE_SSD_BLOCKS',100))
debuginfo.set_level(nv_cache_debug_level)

def nv_cache_log(content):
    debuginfo.info(content)


class FlexKVConnector(KVConnectorBase):

    def __init__(
        self,
        rank: int,
        local_rank: int,
        config: VllmConfig,
    ):
        self.inflight_put = []
        self.config = config
        self.rank = rank
        self.local_rank = local_rank
        nv_cache_log(f"FlexKV init,cpu_blocks:{cpu_blocks}")
        self.first_call = True
        self.disabled = os.getenv('FLEXKV_DISABLED', '0') == '1'

    def select(self, input_tokens: Optional[torch.Tensor],
               roi: Optional[torch.Tensor]) -> List[Optional[torch.Tensor]]:
        """
        assert self.consumer_buffer is not None, "Please initialize the "\
            "consumer buffer before calling select."
        return self.consumer_buffer.drop_select(input_tokens, roi)
        """
        nv_cache_log("SimpleConnector.select called")
        return []

    def insert(self, input_tokens: torch.Tensor, roi: torch.Tensor,
               key: torch.Tensor, value: torch.Tensor,
               hidden: torch.Tensor) -> None:
        """
        assert self.producer_buffer is not None, "Please initialize the "\
            "producer buffer before calling insert."

        self.producer_buffer.insert(input_tokens, roi, key, value, hidden)
        """
        nv_cache_log("SimpleConnector.insert called")

    def get_complete_slot_mapping(self, model_input: "ModelInputForGPUWithSamplingMetadata"):
        # for gpu prefix-cacheing, we need to get complete slot mapping by block table
        if hasattr(model_input.attn_metadata._cached_prefill_metadata, '_cached_prefill_metadata') and \
            model_input.attn_metadata._cached_prefill_metadata._cached_prefill_metadata is not None:
            slot_mapping = model_input.attn_metadata._cached_prefill_metadata._cached_prefill_metadata.slot_mapping
        elif hasattr(model_input.attn_metadata._cached_prefill_metadata, 'slot_mapping') and \
            model_input.attn_metadata._cached_prefill_metadata.slot_mapping is not None:
            slot_mapping = model_input.attn_metadata._cached_prefill_metadata.slot_mapping
        else:
            slot_mapping = model_input.attn_metadata.slot_mapping
        return slot_mapping

    def gen_block_table(self, model_input: "ModelInputForGPUWithSamplingMetadata"):
        slot_mapping = self.get_complete_slot_mapping(model_input)
        block_table = []
        max_len = 0
        for i in range(len(model_input.attn_metadata.seq_lens)):
            slot_mapping_seq = slot_mapping[model_input.attn_metadata.seq_start_loc[i]:model_input.attn_metadata.seq_start_loc[i+1]]
            block_table_seq = slot_mapping_seq[::self.block_size] // self.block_size
            block_table.append(block_table_seq)
            max_len = max(max_len, len(block_table_seq))
        # block_table_seq has different length, so we need to pad them to the same length
        for i in range(len(block_table)):
            block_table[i] = torch.cat([block_table[i], torch.zeros(max_len - len(block_table[i]), dtype=torch.int32).cuda()])
        block_table = torch.stack(block_table).to(torch.int32).cuda()
        model_input.attn_metadata.block_tables = block_table
        if model_input.attn_metadata._cached_prefill_metadata is not None:
            model_input.attn_metadata._cached_prefill_metadata.block_tables = block_table

    def send_kv_caches_and_hidden_states(
        self,
        model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor],
        hidden_or_intermediate_states: Union[torch.Tensor,
                                             IntermediateTensors],
    ) -> None:
        if self.disabled:
            return
        nv_cache_log("SimpleConnector.send_kv_caches_and_hidden_states called, start")
        nv_cache_log(f"kv_caches shape {kv_caches[0].shape} type {kv_caches[0].dtype}")
        slot_mapping = None
        if hasattr(model_input.attn_metadata, '_cached_prefill_metadata') and model_input.attn_metadata._cached_prefill_metadata is not None:
            slot_mapping = model_input.attn_metadata._cached_prefill_metadata.slot_mapping
        nv_cache_log("start NVCache send")
        #if not self.kvpool.is_ready(): #TODO add is_ready function for flexkv
        if self.first_call:
            nv_cache_log("add kv_caches to FlexKV")
            self.block_size = self.config.cache_config.block_size
            model_config = ModelConfig(num_layers=self.config.model_config.get_num_layers(self.config.parallel_config),
                                num_kv_heads=self.config.model_config.get_num_kv_heads(self.config.parallel_config),
                                head_size=self.config.model_config.get_head_size(),
                                element_size=torch.bfloat16.itemsize,
                                use_mla=False,
                                tp_size=1)
            cache_config = CacheConfig(enable_cpu=True,
                                    enable_ssd=False,
                                    enable_remote=False,
                                    use_gds=False,
                                    use_pinned_memory=True,
                                    tokens_per_block=self.block_size,
                                    num_cpu_blocks=cpu_blocks) 
            nv_cache_log(f"model_config {model_config}, cache_config {cache_config}")
            self.kvpool = KVManager(model_config, cache_config, [kv_caches])
            nv_cache_log(f"FlexKV init done")
            self.first_call = False
        # NVCache sync
        # 确保之前的put已经完成了
        for idx, seq_group in enumerate(model_input.sampling_metadata.seq_groups):
            with nvtx.annotate("connector send for loop"):
                if not seq_group.is_prompt:
                    continue
                seq_id = seq_group.seq_ids[0]
                seq_data = seq_group.seq_data[seq_id]

                tokens = torch.tensor(seq_data.get_token_ids(), device="cpu")
                # should mask?
                slot_mapping = self.get_complete_slot_mapping(model_input)
                slot_mapping = slot_mapping[model_input.attn_metadata.seq_start_loc[idx]:model_input.attn_metadata.seq_start_loc[idx+1]]
                ret = self.kvpool.put_async(tokens.detach().cpu(),
                                    token_mask=torch.ones_like(tokens, dtype=torch.bool),
                                    slot_mapping=slot_mapping.clone().cpu().to(torch.int64))
                self.inflight_put.append(ret)
                nv_cache_log(f'request to put {int(tokens.numel())} tokens into NVCACHE')
        nv_cache_log(f'put request finished')
        nv_cache_log("SimpleConnector.send_kv_caches_and_hidden_states called, end")

    def recv_kv_caches_and_hidden_states(
        self, model_executable: torch.nn.Module,
        model_input: "ModelInputForGPUWithSamplingMetadata",
        kv_caches: List[torch.Tensor]
    ) -> Tuple[Union[torch.Tensor, IntermediateTensors], bool,
               "ModelInputForGPUWithSamplingMetadata"]:
        if self.disabled:
            return None, False, model_input
        recv_start_time = time.time()
        nv_cache_log("SimpleConnector.recv_kv_caches_and_hidden_states called, start")
        nv_cache_log(f"kv_caches shape {kv_caches[0].shape} type {kv_caches[0].dtype}")

        if self.first_call:            
            nv_cache_log("add kv_caches to FlexKV")
            self.block_size = self.config.cache_config.block_size
            model_config = ModelConfig(num_layers=self.config.model_config.get_num_layers(self.config.parallel_config),
                                num_kv_heads=self.config.model_config.get_num_kv_heads(self.config.parallel_config),
                                head_size=self.config.model_config.get_head_size(),
                                element_size=torch.bfloat16.itemsize,
                                use_mla=False,
                                tp_size=1)
            cache_config = CacheConfig(enable_cpu=True,
                                    enable_ssd=False,
                                    enable_remote=False,
                                    use_gds=False,
                                    use_pinned_memory=True,
                                    tokens_per_block=self.block_size,
                                    num_cpu_blocks=cpu_blocks)      
            nv_cache_log(f"model_config {model_config}, cache_config {cache_config}")
            self.kvpool = KVManager(model_config, cache_config, [kv_caches])
            nv_cache_log(f"FlexKV init done")
            self.first_call = False
        nv_cache_log("try to recieve from host")
        

        hidden_or_intermediate_states = None
        bypass_model_exec = False

        num_zero_match = 0
        batched_computed_tokens = []
        # NVCache sync
        # 确保之前的put已经完成了
        self.kvpool.wait(self.inflight_put)
        self.inflight_put = []
        slot_mapping_cpu = model_input.attn_metadata.slot_mapping.cpu()
        batch_reqs = []
        batch_seq_lens = []
        batch_num_vllm_cached_tokens = []
        start = time.time()
        for idx, seq_group in enumerate(model_input.sampling_metadata.seq_groups):
            with nvtx.annotate("connector recive for_loop"):
                if not seq_group.is_prompt:
                    num_zero_match += 1
                    continue
                # Prfill只有一个sequence
                # max_batched_num_tokens + chunked prefill 会出bug
                seq_id = seq_group.seq_ids[0]
                seq_data = seq_group.seq_data[seq_id]
                # chunk prefill要匹配所有的而非一片chunk，看看应该选哪个
                seq_len = seq_data.get_len()  # or seq_lens[idx]  #

                tokens = torch.tensor(seq_data.get_token_ids(), device="cpu")
                # nv_cache_log(f"tokens to fetch {tokens}")
                # vllm已经匹配了多少个token
                # 当前vLLM匹配是block alignment的
                num_vllm_cached_tokens = seq_data.get_num_cached_tokens()
                batch_seq_lens.append(seq_len)
                batch_num_vllm_cached_tokens.append(num_vllm_cached_tokens)
                if num_vllm_cached_tokens % self.block_size != 0:
                    raise ValueError(
                        "vLLM cached tokens is not divisible by self.block_size"
                    )
                # Too short to be found in NVCache
                if seq_len - num_vllm_cached_tokens < self.block_size:
                    num_zero_match += 1
                    continue

                # False表示不需要操作（vLLM已匹配），True表示需要操作（vLLM未匹配）
                token_mask = torch.ones_like(tokens, dtype=torch.bool)
                token_mask[:num_vllm_cached_tokens] = False
                # call nvcache get-------------------------------------
                # 完整的token list和全部的kv caches
                nv_cache_log(f"recieved {num_vllm_cached_tokens} tokens from vllm prefix cache")
                nv_cache_log(f"trying to recieve {int(token_mask.sum())} tokens from NVCACHE")
                slot_mapping = slot_mapping_cpu[model_input.attn_metadata.query_start_loc[idx]:model_input.attn_metadata.query_start_loc[idx+1]]

                # breakpoint()
                req =  self.kvpool.get_async(tokens,
                                    token_mask=token_mask,
                                    slot_mapping=slot_mapping,
                                    layer_granularity=-1)
                batch_reqs.append(req)
        masks = self.kvpool.wait(batch_reqs)
        end = time.time()
        sum_nvc_cached_tokens = 0
        for i, req in enumerate(batch_reqs):
            seq_len = batch_seq_lens[i]
            num_nvc_cached_tokens = torch.sum(masks[req]).item()
            num_vllm_cached_tokens = batch_num_vllm_cached_tokens[i]
            num_total_cached_tokens = num_vllm_cached_tokens + num_nvc_cached_tokens
            sum_nvc_cached_tokens += num_nvc_cached_tokens
            nv_cache_log(f"seq {i} recieved {num_nvc_cached_tokens} tokens from NVCACHE")
            if num_total_cached_tokens % self.block_size != 0:
                raise ValueError(
                    "Nvcache cached tokens is not divisible by self.block_size"
                )
            # 如果完全匹配计算可能会出问题, bypass 逻辑？
            if num_total_cached_tokens == seq_len:
                num_total_cached_tokens -= self.block_size
                sum_nvc_cached_tokens -= self.block_size
            if num_total_cached_tokens == 0:
                num_zero_match += 1
            nv_cache_log(f"seq {i} matched {num_total_cached_tokens} tokens in total.")
            batched_computed_tokens.append(num_nvc_cached_tokens)

        nv_cache_log(f'recieved {sum_nvc_cached_tokens} tokens from NVCACHE')
        nv_cache_log(f"recieve cost time: {(end-start)*1000:.2f} ms")
        # end call-----------------------------------------------

        # 并不是所有的都没匹配到，需要重构model input
        if num_zero_match < len(model_input.attn_metadata.query_start_loc) - 1:
            # need to confirm: num_computed_tokens need to be align with vllm block.
            # need to confirm: directly pass slot mapping to atten metadata.
            # need to confirm: is the blocktable well handled during kv-cache recieve
            # need to confirm: conditions to jump in recv_kv_cache
            # need to confirm: conditions that seqgroup have multiple seqs inside.
            with nvtx.annotate("connector rebuild input"):
                # TODO: reduce the time of rebuild input
                model_input = build_new_prefill_input(
                    model_input, batched_computed_tokens, kv_caches[0][0].device
                )
                self.gen_block_table(model_input)
                logger.debug("Rebuild the input!")
        nv_cache_log("SimpleConnector.recv_kv_caches_and_hidden_states called, end")
        recv_end_time = time.time()
        nv_cache_log(f"recv cost time e2e: {(recv_end_time-recv_start_time)*1000:.2f} ms")
        return hidden_or_intermediate_states, bypass_model_exec, model_input

    def close(self):
        nv_cache_log("SimpleConnector.close called")
        nv_cache_log(f"waiting {len(self.inflight_put)} put request to finish")
        self.kvpool.wait(self.inflight_put)
        nv_cache_log(f"kvpool shutdown")
        self.kvpool.shutdown()
        nv_cache_log(f"kvpool shutdown done")
        self.producer_data_pipe.close()
        self.consumer_data_pipe.close()
        if self.config.kv_connector == "PyNcclConnector":
            self.producer_signal_pipe.close()
            self.consumer_signal_pipe.close()
        elif self.config.kv_connector == "MooncakeConnector":
            # MooncakePipe reuses data_pipe for signal_pipe, so we only have to
            # close the data_pipe.
            pass

    def __del__(self):
        self.kvpool.shutdown()
