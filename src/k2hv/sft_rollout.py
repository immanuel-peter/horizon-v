"""Single-turn image SFT preparation for Miles, with answer-only supervision."""

PROCESSOR = None


def prepare_sample(processor, sample):
    messages = sample.prompt
    if len(messages) != 2 or messages[0]["role"] != "user" or messages[1]["role"] != "assistant":
        raise ValueError("Gate SFT supports one user/assistant turn")
    if messages[1].get("think") != "" or not isinstance(messages[1]["content"], str):
        raise ValueError("Gate SFT requires a string answer and explicit empty think")
    text = processor.apply_chat_template(messages, tokenize=False)
    output = processor(text=text, images=sample.multimodal_inputs["images"], return_tensors="pt")
    ids = output["input_ids"][0].tolist()
    tokenizer = processor.tokenizer
    closing = tokenizer.encode("</ifm|think>", add_special_tokens=False)
    ends = [i + len(closing) for i in range(len(ids) - len(closing) + 1) if ids[i:i + len(closing)] == closing]
    if len(ends) != 1:
        raise ValueError("Expected exactly one closing think marker")
    start = ends[0]
    supervised = tokenizer.decode(ids[start:], skip_special_tokens=False)
    expected = messages[1]["content"] + "<|ifm|im_end|>"
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
    return samples
