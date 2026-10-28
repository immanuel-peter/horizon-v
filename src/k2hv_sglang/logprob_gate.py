"""Sequential HF reference capture and native SGLang teacher-forced comparison."""
import argparse
import base64
import io
import json
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from .smoke import post

# BF16 log-softmax accumulates output-logit rounding across different kernels.
# These bounds are declared before the first native comparison and are not auto-tuned.
MAX_ERROR = 0.2
MEAN_ERROR = 0.03


def capture(args):
    processor = AutoProcessor.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    cases = []
    for name, prompt in (("short", "The capital of France is Paris. The capital of Japan is Tokyo."), ("long", "Explain a sequence of integers. " * 128)):
        cases.append((name, processor.tokenizer(prompt, return_tensors="pt", add_special_tokens=False), []))
    for size in ((448, 448), (560, 112), (112, 560), (1, 1)):
        image = Image.new("RGB", size, (70, 120, 180))
        output = io.BytesIO()
        image.save(output, format="PNG")
        encoded = base64.b64encode(output.getvalue()).decode()
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Describe this image."}]}, {"role": "assistant", "think": "", "content": "A blue image."}]
        inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt")
        cases.append((f"image-{size[0]}x{size[1]}", inputs, [encoded]))
    references = []
    for name, inputs, images in cases:
        ids = inputs["input_ids"][0]
        with torch.no_grad():
            logits = model(**{k: v.cuda() for k, v in inputs.items()}, use_cache=False).logits[0, :-1].float()
            logprobs = logits.log_softmax(-1).gather(-1, ids[1:].cuda().unsqueeze(-1)).squeeze(-1).cpu().tolist()
        references.append({"name": name, "ids": ids.tolist(), "image_data": images, "logprobs": logprobs})
    args.reference.write_text(json.dumps({"seed": 0, "dtype": "bfloat16", "backend": "sdpa", "max_error": MAX_ERROR, "mean_error": MEAN_ERROR, "cases": references}))


def evaluate(args):
    references = json.loads(args.reference.read_text())
    results = []
    for case in references["cases"]:
        payload = {"input_ids": case["ids"], "sampling_params": {"max_new_tokens": 1, "temperature": 0}, "return_logprob": True, "logprob_start_len": 0}
        if case["image_data"]:
            payload["image_data"] = case["image_data"]
        result = post(args.url, "/generate", payload)
        rows = result["meta_info"]["input_token_logprobs"]
        assert len(rows) == len(case["ids"]), (len(rows), len(case["ids"]))
        errors = []
        for position, (row, token) in enumerate(zip(rows, case["ids"])):
            if position == 0 or token == 250062:
                continue
            assert row[1] == token, (position, row[1], token)
            assert row[0] is not None, position
            errors.append(abs(row[0] - case["logprobs"][position - 1]))
        maximum, mean = max(errors), sum(errors) / len(errors)
        results.append({"name": case["name"], "input_length": len(case["ids"]), "compared_positions": len(errors), "max_error": maximum, "mean_error": mean, "status": "passed" if maximum <= MAX_ERROR and mean <= MEAN_ERROR else "failed"})
    report = {"max_error_tolerance": MAX_ERROR, "mean_error_tolerance": MEAN_ERROR, "results": results}
    args.report.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if any(row["status"] != "passed" for row in results):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("capture", "evaluate"))
    parser.add_argument("--checkpoint", default="hf/K2-Horizon-V-init")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--report", type=Path, default=Path("outputs/step04/logprob.json"))
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    args = parser.parse_args()
    (capture if args.stage == "capture" else evaluate)(args)


if __name__ == "__main__":
    main()
