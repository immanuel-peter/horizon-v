import hashlib
import os

import torch

PROCESSOR = None


def prepare_sample(processor, sample):
    messages = sample.prompt
    if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        raise ValueError("Gate SFT supports one user/assistant turn")
    answer = messages[1]["content"]
    if isinstance(answer, list) and all(p.get("type") == "text" for p in answer):
        answer = "".join(p["text"] for p in answer)
    if messages[1].get("think") != "" or not isinstance(answer, str):
        raise ValueError("Gate SFT requires a string answer and explicit empty think")
    text = processor.apply_chat_template(messages, tokenize=False)
    output = processor(text=text, images=sample.multimodal_inputs["images"], return_tensors="pt")
    ids = output["input_ids"][0].tolist()
    tokenizer = processor.tokenizer
    closing = tokenizer.encode("</ifm|think>", add_special_tokens=False)
    ends = [i + len(closing) for i in range(len(ids) - len(closing) + 1) if ids[i : i + len(closing)] == closing]
    if len(ends) != 1:
        raise ValueError("Expected exactly one closing think marker")
    start = ends[0]
    supervised = tokenizer.decode(ids[start:], skip_special_tokens=False)
    expected = answer + "<|ifm|im_end|>"
    if supervised != expected:
        raise ValueError(f"Unexpected supervised span: {supervised!r}")
    sample.tokens = ids
    sample.response_length = len(ids) - start
    sample.loss_mask = [1] * sample.response_length
    sample.multimodal_train_inputs = {k: output[k] for k in ("pixel_values", "image_grid_thw")}
    sample.reward = 0
    return sample


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    if evaluation:
        raise ValueError("SFT rollout is training-only")
    global PROCESSOR
    if PROCESSOR is None:
        from miles.utils.processing_utils import load_processor

        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)
    samples = data_buffer.get_samples(args.rollout_batch_size)
    for (sample,) in samples:
        prepare_sample(PROCESSOR, sample)

        if os.environ.get("K2HV_MILES_GATE"):
            from pathlib import Path

            root = Path(os.environ["K2HV_MILES_GATE"]) / "fixture"
            root.mkdir(parents=True, exist_ok=True)
            pixels = sample.multimodal_train_inputs["pixel_values"]
            key = hashlib.sha256(pixels.numpy().tobytes()).hexdigest()
            path = root / (key + ".pt")
            if not path.exists():
                torch.save({"tokens": sample.tokens, "response_length": sample.response_length, "loss_mask": sample.loss_mask, **sample.multimodal_train_inputs}, path)
    return samples
