import asyncio
from functools import partial

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem, MultimodalProcessorOutput
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor, MultimodalSpecialTokens

from .gate_trace import emit
from .input_layout import expand_image_tokens, image_hash, split_image_features
from .k2_horizon_v import K2HorizonVForConditionalGeneration


class K2HorizonVImageProcessor(BaseMultimodalProcessor):
    models = [K2HorizonVForConditionalGeneration]
    gpu_image_decode = False
    prefer_tokenized_input = True
    preserve_processor_input_ids = True
    supports_mm_processor_concurrency = False

    def __init__(self, hf_config, server_args, _processor, transport_mode, *args, **kwargs):
        if _processor.image_processor.backend != "pil":
            raise ValueError("Use --image-processor-backend pil to match the HF processor")
        for name in ("image_token_id", "image_start_token_id", "image_end_token_id"):
            if getattr(_processor, name) != getattr(hf_config, name):
                raise ValueError(f"HF config and processor disagree on {name}")
        super().__init__(hf_config, server_args, _processor, transport_mode, *args, **kwargs)
        if set(self.image_config) - {"min_image_tokens", "max_image_tokens"}:
            raise ValueError("Only min_image_tokens/max_image_tokens overrides are supported in the image process config")

    def validate_mm_input_ids(self, input_ids, request_obj):
        if input_ids is None or getattr(request_obj, "image_data", None):
            return
        ids = input_ids.tolist() if isinstance(input_ids, torch.Tensor) else input_ids
        image_ids = {self.hf_config.image_start_token_id, self.hf_config.image_token_id, self.hf_config.image_end_token_id}
        if any(token in image_ids for token in ids):
            raise ValueError("Image marker tokens require image_data")

    async def process_mm_data_async(self, image_data, input_text, request_obj, *args, **kwargs):
        if getattr(request_obj, "video_data", None) or getattr(request_obj, "audio_data", None):
            raise ValueError("K2-Horizon-V supports images and text only")
        if isinstance(input_text, str):
            ids = self._tokenizer.encode(input_text, add_special_tokens=False)
        elif isinstance(input_text, torch.Tensor):
            if input_text.ndim != 1:
                raise ValueError("A multimodal request must contain one token sequence")
            ids = input_text.tolist()
        else:
            ids = list(input_text)

        loaded = await self.fast_load_mm_data(prompt="", multimodal_tokens=MultimodalSpecialTokens(image_token_id=self.hf_config.image_token_id), image_data=image_data or [])
        if any(isinstance(image, (dict, list)) for image in loaded.images):
            raise ValueError("Precomputed embeddings, processor-output dictionaries and image frame lists are not supported")
        if loaded.images:
            loop = asyncio.get_running_loop()
            encoded = await loop.run_in_executor(self.io_executor, partial(self._processor.image_processor, loaded.images, return_tensors="pt", **self.image_config))
            grids = encoded["image_grid_thw"]
            features = split_image_features(encoded["pixel_values"], grids, merge_size=self._processor.image_processor.merge_size)
        else:
            grids, features = torch.empty((0, 3), dtype=torch.long), []
        ids, offsets = expand_image_tokens(
            ids, grids, image_start_id=self.hf_config.image_start_token_id, image_token_id=self.hf_config.image_token_id,
            image_end_id=self.hf_config.image_end_token_id, merge_size=self._processor.image_processor.merge_size,
        )
        items = []
        for (pixels, grid), offset in zip(features, offsets):
            item = MultimodalDataItem(modality=Modality.IMAGE, feature=pixels, offsets=[offset], model_specific_data={"image_grid_thw": grid})
            item.set_hash(image_hash(pixels, grid))
            items.append(item)
        emit("processor", ids=ids, offsets=offsets, pixels=[p for p, g in features], grids=grids)
        items = self._finalize_mm_items(items, images=loaded.images)
        return MultimodalProcessorOutput(mm_items=items, input_ids=ids, im_token_id=self.hf_config.image_token_id,
                                         im_start_id=self.hf_config.image_start_token_id, im_end_id=self.hf_config.image_end_token_id)
