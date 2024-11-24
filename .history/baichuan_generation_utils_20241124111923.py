from typing import List
import torch

def build_chat_input(model, tokenizer, messages: List[dict], max_new_tokens: int=0):
    def _parse_messages(messages, split_role="user"):
        system, rounds = "", []
        round = []
        for i, message in enumerate(messages):
            if message["role"] == "system":
                assert i == 0
                system = message["content"]
                continue
            if message["role"] == split_role and round:
                rounds.append(round)
                round = []
            round.append(message)
        if round:
            rounds.append(round)
        return system, rounds
    max_new_tokens = max_new_tokens or model.generation_config.max_new_tokens
    max_input_tokens = model.config.model_max_length - max_new_tokens
    system, rounds = _parse_messages(messages, split_role="user")
    system_tokens = tokenizer.encode(system)
    max_history_tokens = max_input_tokens - len(system_tokens)

    history_tokens = []
    raw_history_tokens = ""
    for round in rounds[::-1]:
        round_tokens = []
        raw_round_tokens = ""
        for message in round:
            if message["role"] == "user":
                round_tokens.append(model.generation_config.user_token_id)
                raw_round_tokens += tokenizer._convert_id_to_token(model.generation_config.user_token_id)
            else:
                round_tokens.append(model.generation_config.assistant_token_id)
                raw_round_tokens += tokenizer._convert_id_to_token(model.generation_config.assistant_token_id)
            round_tokens.extend(tokenizer.encode(message["content"]))
            raw_round_tokens += message["content"]
        if len(history_tokens) == 0 or len(history_tokens) + len(round_tokens) <= max_history_tokens:
            history_tokens = round_tokens + history_tokens  # concat left
            raw_history_tokens = raw_round_tokens + raw_history_tokens
            if len(history_tokens) < max_history_tokens:
                continue
        break

    input_tokens = system_tokens + history_tokens
    raw_input_tokens = system + raw_history_tokens
    if messages[-1]["role"] != "assistant":
        input_tokens.append(model.generation_config.assistant_token_id)
        raw_input_tokens += tokenizer._convert_id_to_token(model.generation_config.assistant_token_id)
    input_tokens = input_tokens[-max_input_tokens:]  # truncate left
    return raw_input_tokens, torch.LongTensor([input_tokens]).to(model.device)