import math
import torch

def resize_mask(mask, H, W):
    return torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(H, W), mode="nearest").squeeze(0).squeeze(0)


def find_inner_sentence_token_span_qwen(tokenizer, full_prompt: str, inner_sentence: str, max_sequence_length: int = 512):
    """
    Returns (start_idx, end_idx) token span for `inner_sentence` inside `full_prompt` for Qwen.
    Uses `apply_chat_template` logic matching `_get_qwen3_prompt_embeds`.
    Indices are over the token dimension used by the model.
    end_idx is exclusive. Returns None if no contiguous match is found.
    """
    messages = [{"role": "user", "content": full_prompt}]

    if hasattr(tokenizer, "apply_chat_template"):
        try:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
    else:
        raise ValueError("Tokenizer must have apply_chat_template method")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_sequence_length,
    )

    full_ids = inputs["input_ids"][0]
    full_mask = inputs["attention_mask"][0]
    valid_len = int(full_mask.sum().item())
    full_ids = full_ids[:valid_len].tolist()

    tokenizer_obj = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
    inner_ids = tokenizer_obj.encode(inner_sentence, add_special_tokens=False)

    if not inner_ids:
        return None, None

    n = len(inner_ids)
    for i in range(0, len(full_ids) - n + 1):
        if full_ids[i : i + n] == inner_ids:
            return i, i + n

    return None, None


def create_position_mask_list(instance_bboxes_xyxy_normalized, height, width, vae_scale_factor):
    position_mask_list = []
    for i in range(len(instance_bboxes_xyxy_normalized)):
        instance_box_i = instance_bboxes_xyxy_normalized[i]
        position_mask_list.append(change_box_to_position_mask(instance_box_i, height // vae_scale_factor // 2, width // vae_scale_factor // 2))
    return position_mask_list


def change_box_to_position_mask(box, H, W):
    x1, y1, x2, y2 = box
    x1 = math.floor(x1 * W)
    y1 = math.floor(y1 * H)
    x2 = math.ceil(x2 * W)
    y2 = math.ceil(y2 * H)
    position_mask = torch.zeros(H, W)
    position_mask[y1: y2, x1: x2] = 1
    return position_mask
