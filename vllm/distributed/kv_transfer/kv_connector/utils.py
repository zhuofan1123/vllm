from vllm.worker.model_runner import ModelInputForGPUWithSamplingMetadata
from vllm.attention.backends.flash_attn import FlashAttentionMetadata
import torch
from copy import deepcopy
def build_new_prefill_input(
    model_input: "ModelInputForGPUWithSamplingMetadata",
    num_computed_tokens: list,
    device: torch.device,
) -> "ModelInputForGPUWithSamplingMetadata":
    ''':
    model_input: the original ModelInputForGPUWithSamplingMetadata object
    num_computed_tokens: number of tokens filled in immutable kv-blocks,
    '''
    assert model_input.attn_metadata is not None
    assert isinstance(model_input.attn_metadata, FlashAttentionMetadata)
    assert model_input.attn_metadata.context_lens_tensor is not None
    assert model_input.attn_metadata.block_tables is not None
    assert model_input.attn_metadata.query_start_loc is not None
    assert model_input.input_positions is not None
    assert len(num_computed_tokens)== len(model_input.seq_lens), "the lenth of num_computed_tokens should be equal to the batch_size"
    print(f"model_input.query_lens: {model_input.query_lens}, num_computed_tokens: {num_computed_tokens}, seq_lens: {model_input.seq_lens}")
    new_query_lens = [model_input.query_lens[i] - num_computed_tokens[i] for i in range(len(model_input.seq_lens))]
    attn_metadata = deepcopy(model_input.attn_metadata)
    sampling_metadata = deepcopy(model_input.sampling_metadata)
    new_input_tokens_list = []
    new_input_positions_list = []
    new_slot_mapping_list = []
    new_selected_token_indices_list = []
    new_query_start_loc_list = [0]
    total_query_len = 0
    end_pos = 0
    for ind,new_query_len in enumerate(new_query_lens):
        total_query_len += new_query_len
        new_query_start_loc_list.append(total_query_len)
        new_selected_token_indices_list.append(total_query_len-1)
        start_pos = end_pos + num_computed_tokens[ind]
        end_pos = start_pos + new_query_len
        
        new_input_tokens_list.append(
            model_input.input_tokens[start_pos:end_pos].clone()
        )
        new_input_positions_list.append(
            model_input.input_positions[start_pos:end_pos].clone()
        )
        new_slot_mapping_list.append(
            attn_metadata.slot_mapping[start_pos:end_pos].clone()
        )
        sampling_metadata.seq_groups[ind].query_len = new_query_len

    
    new_input_tokens = torch.cat(new_input_tokens_list)
    new_input_positions = torch.cat(new_input_positions_list)
    attn_metadata.slot_mapping = torch.cat(new_slot_mapping_list)

    
    
    
    attn_metadata.num_prefill_tokens = sum(new_query_lens)
    attn_metadata.context_lens_tensor = torch.tensor(num_computed_tokens, device=device, dtype=torch.int32)
    attn_metadata.max_query_len = max(new_query_lens)
    attn_metadata.query_start_loc = torch.tensor(new_query_start_loc_list, device=device, dtype=torch.int32)

    attn_metadata._cached_prefill_metadata = deepcopy(attn_metadata)
    
    sampling_metadata.selected_token_indices = torch.tensor(new_selected_token_indices_list, device=device)

    new_model_input = ModelInputForGPUWithSamplingMetadata(
        input_tokens=new_input_tokens,
        input_positions=new_input_positions,
        seq_lens=model_input.seq_lens,
        query_lens=new_query_lens,
        lora_mapping=model_input.lora_mapping,
        lora_requests=model_input.lora_requests,
        attn_metadata=attn_metadata,
        prompt_adapter_mapping=model_input.prompt_adapter_mapping,
        prompt_adapter_requests=model_input.prompt_adapter_requests,
        multi_modal_kwargs=model_input.multi_modal_kwargs,
        request_ids_to_seq_ids=model_input.request_ids_to_seq_ids,
        finished_requests_ids=model_input.finished_requests_ids,
        virtual_engine=model_input.virtual_engine,
        sampling_metadata=sampling_metadata,
        is_prompt=model_input.is_prompt,
        async_callback=model_input.async_callback,
    )
    # breakpoint()
    return new_model_input
