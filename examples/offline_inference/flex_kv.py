# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of cpu offloading
with LMCache.

Note that `pip install lmcache` is needed to run this example.
Learn more about LMCache in https://github.com/LMCache/LMCache.
"""
import os
import time

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    # This example script runs two requests with a shared prefix.
    shared_prompt_1k = "You are a helpful assistant. " * 160
    prompts = [
        # request 0
        [shared_prompt_1k],
        # request 1
        [shared_prompt_1k + "How are you? ", 
         shared_prompt_1k + "Who are you? "],
        # request 2
        [shared_prompt_1k * 2],
        # request 3
        [shared_prompt_1k * 2 + "Tell me a very long story", 
         shared_prompt_1k * 2 + "Tell me a very short story"],
        # request 4
        [shared_prompt_1k * 3],
        # request 5
        [shared_prompt_1k * 3 + "What is the capital of France?", 
         shared_prompt_1k * 3 + "What is the capital of China?",
         shared_prompt_1k * 3 + "How to make a cake?", 
         shared_prompt_1k * 3 + "How to make a pizza"]
    ]
    

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    ktc = KVTransferConfig.from_cli(
        '{"kv_connector":"FlexKVConnector", "kv_role":"kv_both"}')
    # Set GPU memory utilization to 0.8 for an A40 GPU with 40GB
    # memory. Reduce the value if your GPU has less memory.
    # Note that LMCache is not compatible with chunked prefill for now.
    llm = LLM(model="Qwen/Qwen3-8B",
            kv_transfer_config=ktc,
            max_model_len=10000,
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
    for i in range(len(ttft_list)):
        for j in range(len(ttft_list[i])):
            print(f"Request {i} Sequence {j} Length {seq_len_list[i][j]/1000:.3f}k TTFT: {ttft_list[i][j]:3f}s")
