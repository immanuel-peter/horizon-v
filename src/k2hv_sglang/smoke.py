"""Exercise an already-running SGLang server with text and image requests."""

import argparse
import base64
import io
import json
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image
from transformers import AutoProcessor

from .runtime import SGLANG_REVISION

IMAGE_SIZE = 448
IMAGE_TOKENS = 256


def post(base_url, path, payload):
    request = Request(base_url.rstrip("/") + path, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=300) as response:
        body = response.read()
    if not body:
        return None
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return {"text": body.decode()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    picture = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), color=(70, 120, 180))
    image_file = io.BytesIO()
    picture.save(image_file, format="PNG")
    encoded = base64.b64encode(image_file.getvalue()).decode()
    messages = [{"role": "user", "content": [{"type": "image", "image": picture}, {"type": "text", "text": "Describe this image."}]}]
    inputs = processor.apply_chat_template(messages, tokenize=True, return_dict=True, return_tensors="pt", add_generation_prompt=True)
    ids = inputs["input_ids"][0].tolist()
    if ids.count(processor.image_token_id) != IMAGE_TOKENS:
        raise RuntimeError(f"HF processor produced {ids.count(processor.image_token_id)} image tokens, expected {IMAGE_TOKENS}")
    results = {
        "text": post(args.url, "/generate", {"text": "The capital of France is", "sampling_params": {"max_new_tokens": 2, "temperature": 0}}),
        "miles_image": post(args.url, "/generate", {"input_ids": ids, "image_data": [encoded], "sampling_params": {"max_new_tokens": 2, "temperature": 0}}),
        "chat_image": post(args.url, "/v1/chat/completions", {"model": str(args.model_path), "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + encoded}}, {"type": "text", "text": "Describe this image."}]}],
            "max_tokens": 2, "temperature": 0}),
    }
    if not isinstance(results["text"].get("text"), str):
        raise RuntimeError("Text request returned no text")
    if not isinstance(results["miles_image"].get("text"), str):
        raise RuntimeError("Pre-expanded image request returned no text")
    if results["miles_image"]["meta_info"]["prompt_tokens"] != len(ids):
        raise RuntimeError(f"Server counted {results['miles_image']['meta_info']['prompt_tokens']} prompt tokens, expected {len(ids)}")
    if not results["chat_image"].get("choices"):
        raise RuntimeError("Image chat request returned no choices")
    results["reference_revision"] = SGLANG_REVISION
    results["hf_image_tokens"] = IMAGE_TOKENS
    results["hf_prompt_tokens"] = len(ids)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
