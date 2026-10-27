PROCESSOR_AUTO_MAP = {"AutoProcessor": "processing_k2_horizon_v.K2HorizonVProcessor"}
MAX_IMAGE_TOKENS = 1024
IMAGE_TEMPLATE_BRANCH = """        {%- elif part is mapping and part.type in ['image', 'image_url'] and role == 'user' -%}
{{- '<image>reserved_special_token_12</image>' -}}
"""


def patch_chat_template(source: str) -> str:
    anchor = "{{- part.text -}}\n        {%- else -%}"
    if source.count(anchor) != 1:
        raise ValueError("Expected one K2 text-content branch in the source chat template")
    return source.replace(anchor, "{{- part.text -}}\n" + IMAGE_TEMPLATE_BRANCH + "        {%- else -%}")


def processor_image_config(source: dict) -> dict:
    return {
        **source,
        "image_processor_type": "Glm5NextImageProcessorPil",
        "processor_class": "K2HorizonVProcessor",
        "max_image_tokens": MAX_IMAGE_TOKENS,
        "auto_map": PROCESSOR_AUTO_MAP,
    }
