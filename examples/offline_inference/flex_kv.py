# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of cpu offloading
with LMCache.

Note that `pip install lmcache` is needed to run this example.
Learn more about LMCache in https://github.com/LMCache/LMCache.
"""
import os
import time
import argparse

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # 接收参数：--seq-len 1000 --cache-ratio 0.5
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1000)
    parser.add_argument("--cache-ratio", type=float, default=1.0)
    args = parser.parse_args()


    base_prompt = "You are a helpful assistant. "  # 6 tokens
    cached_prompt = int(args.seq_len * args.cache_ratio) // 6 * base_prompt
    prefill_prompt = (args.seq_len // 6) * base_prompt
    prompts = [
        # request 0 for warmup
        [cached_prompt],
        # request 1
        [prefill_prompt + "What is the capital of China? "],
    ]
    
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"FlexKVConnector", "kv_role":"kv_both"}')
    # Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
    # memory. Reduce the value if your GPU has less memory.
    # Note that LMCache is not compatible with chunked prefill for now.
    llm = LLM(model="Qwen/Qwen3-32B",
            kv_transfer_config=ktc,
            max_model_len=20000,
            enable_chunked_prefill=False,
            enforce_eager=True,
            block_size=16,
            enable_prefix_caching=False,
            gpu_memory_utilization=0.8)
    ttft_list = []
    seq_len_list = []
    for i, prompt in enumerate(prompts):
        ttft_list.append([])
        seq_len_list.append([])
        outputs = llm.generate(prompt, sampling_params)
        for j, output in enumerate(outputs):
            generated_text = output.outputs[0].text
            ttft = output.metrics.first_token_time - output.metrics.first_scheduled_time
            ttft_list[i].append(ttft)
            seq_len_list[i].append(len(output.prompt_token_ids))
            print(f"Generated text: {generated_text!r}")
        print(f"Request {i} done.")
    print(f"Cache ratio: {args.cache_ratio}")
    for i in range(len(ttft_list)):
        for j in range(len(ttft_list[i])):
            print(f"Request {i} Sequence {j} Length {seq_len_list[i][j]/1000:.3f}k TTFT: {ttft_list[i][j]:3f}s")
