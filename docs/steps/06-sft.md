# Step 6: Stage 1, projector SFT

Train only the projector on short-answer image data, in upstream Miles with the FSDP backend and no rollout engines.

## Custom SFT rollout

Miles's stock `miles/rollout/sft_rollout.py` is text-only. It tokenizes with `tokenizer.apply_chat_template`, which leaves one pad token per image, and it never sets `sample.multimodal_train_inputs`. Run as-is, SFT would train the projector on text with no images. Replace it:

```python
# k2hv/sft_rollout.py
from miles.utils.processing_utils import load_processor

PROCESSOR = None
THINK_CLOSE_ID = None


def generate_rollout(args, rollout_id, data_buffer, evaluation=False):
    global PROCESSOR, THINK_CLOSE_ID
    if PROCESSOR is None:
        PROCESSOR = load_processor(args.hf_checkpoint, trust_remote_code=True)
        THINK_CLOSE_ID = PROCESSOR.tokenizer.convert_tokens_to_ids("</ifm|think>")

    samples = data_buffer.get_samples(args.rollout_batch_size)
    for (s,) in samples:
        text = PROCESSOR.apply_chat_template(s.prompt, tokenize=False)
        out = PROCESSOR(text=text, images=s.multimodal_inputs["images"], return_tensors="pt")
        ids = out["input_ids"][0].tolist()
        start = len(ids) - ids[::-1].index(THINK_CLOSE_ID)
        s.tokens = ids
        s.response_length = len(ids) - start
        s.loss_mask = [1] * s.response_length
        s.multimodal_train_inputs = {
            "pixel_values": out["pixel_values"],
            "image_grid_thw": out["image_grid_thw"],
        }
        s.reward = 0
    return samples
```

Single-turn only. Multi-turn data would need one masked span per assistant turn.

### Loss mask: answer only

The loss covers the answer and `<|ifm|im_end|>`, the tokens after `</ifm|think>`. If the empty think block `<ifm|think>\n</ifm|think>` were in the loss, the projector would learn image embeddings that make K2 close its thinking immediately. That works directly against [stage 2](08-rl.md), where K2 must reason about images.

### No `--apply-chat-template`

With that flag, Miles turns `sample.prompt` into a generation-prompt string (`miles/utils/data.py:228`) and the assistant turn is lost. With `--multimodal-keys` set and no `--apply-chat-template`, Miles keeps the message list and fills `sample.multimodal_inputs` with `{"images": [...]}`.

## Launch

Adapted from `examples/geo3k_vlm/run_geo3k_vlm_sft.sh`. Run on a GPU node with the project venv active ([step 0](00-environment.md)):

```bash
python third_party/miles/train_async.py \
  --train-backend fsdp --gradient-checkpointing --attn-implementation flash_attention_2 \
  --actor-num-nodes 1 --actor-num-gpus-per-node 8 \
  --hf-checkpoint /models/K2-Horizon-V-init --load /models/K2-Horizon-V-init \
  --rollout-function-path k2hv.sft_rollout.generate_rollout \
  --prompt-data /data/sft.parquet --input-key messages --multimodal-keys '{"image": "images"}' \
  --rollout-shuffle --num-epoch 1 --rollout-batch-size 256 --global-batch-size 256 \
  --loss-type sft_loss --calculate-per-token-loss \
  --disable-compute-advantages-and-returns --debug-train-only \
  --optimizer adam --lr 1e-3 --min-lr 1e-5 --lr-decay-style cosine \
  --lr-warmup-fraction 0.03 --weight-decay 0 \
  --save /ckpt/sft --save-interval 500
```

- Use whichever attention implementation K2's remote code supports.
- `--debug-train-only` starts no SGLang engines.
- `--loss-mask-type` does not matter here: the custom rollout builds the mask.

## Hyperparameters

| Setting | Value | Note |
|---|---|---|
| LR | 1e-3, cosine to 1e-5 | LLaVA's projector-pretraining LR. Sweep {2e-4, 1e-3} on a 50k-sample slice first. |
| Warmup | 3% | |
| Batch | 256 | |
| Weight decay | 0 | |
| Epochs | 1 | |

## Hardware and cost

- One 8×H100 node is more than enough. The frozen weights take a few GB per GPU when sharded; the optimizer state covers only ~33.6M params.
- About 0.4B tokens through an 8B model at roughly three forward-pass equivalents per step (forward, backward without weight gradients, checkpoint recompute): about 2e19 FLOPs, or roughly 15 H100-hours at 40% MFU.

## Export to HF format

```bash
python third_party/miles/tools/convert_fsdp_to_hf.py \
  --input-dir /ckpt/sft/<iter> \
  --output-dir /models/K2-Horizon-V-sft \
  --origin-hf-dir /models/K2-Horizon-V-init
```

`--origin-hf-dir` copies every non-weight file from the init repo: code, tokenizer, template and processor configs. The SFT checkpoint gets the same layout as the init checkpoint.
