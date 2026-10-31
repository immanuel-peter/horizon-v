"""Image decoding, content-addressed storage, perceptual hashes and model-input geometry."""

import hashlib
import io
import math
import os
import warnings
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = 200_000_000
FORMATS = {"JPEG": "jpg", "PNG": "png", "WEBP": "webp", "GIF": "gif", "BMP": "bmp", "TIFF": "tiff", "MPO": "jpg"}
MODES = {"RGB", "RGBA", "L", "LA", "P", "PA", "CMYK", "1", "YCbCr"}
THUMB = 32

# qwen_vl_utils 0.0.14 defaults for image_patch_size=14, which Miles passes for this processor
QWEN_FACTOR = 28
QWEN_MIN_PIXELS = 4 * QWEN_FACTOR**2
QWEN_MAX_PIXELS = 16384 * QWEN_FACTOR**2
QWEN_MAX_RATIO = 200


class ImageError(ValueError):
    pass


def find_asset(root, sha):
    matches = list((Path(root) / "images" / sha[:2]).glob(sha + ".*"))
    return matches[0] if matches else None


def decode(data):
    """Fully decode bytes; return (image, format) or raise ImageError with a stable reason."""
    if not data:
        raise ImageError("image_missing")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            image = Image.open(io.BytesIO(data))
            fmt = image.format
            image.load()
    except Image.DecompressionBombError as e:
        raise ImageError("image_too_large") from e
    except Exception as e:
        raise ImageError("image_corrupt") from e
    if fmt not in FORMATS:
        raise ImageError(f"image_format_{str(fmt).lower()}")
    if image.mode not in MODES:
        raise ImageError(f"image_mode_{image.mode}")
    if min(image.size) < 8:
        raise ImageError("image_too_small")
    return image, fmt


def has_unhandled_alpha(image):
    """Transparency that qwen_vl_utils.to_rgb would drop instead of compositing on white."""
    if image.mode in ("LA", "PA"):
        return True
    return image.mode == "P" and "transparency" in image.info


def normalize_alpha(image):
    """Apply qwen_vl_utils' RGBA rule (white background) to modes it would otherwise flatten wrongly."""
    rgba = image.convert("RGBA")
    background = Image.new("RGB", rgba.size, (255, 255, 255))
    background.paste(rgba, mask=rgba.split()[3])
    out = io.BytesIO()
    background.save(out, format="PNG", optimize=False)
    return out.getvalue()


def to_rgb(image):
    if image.mode == "RGBA":
        background = Image.new("RGB", image.size, (255, 255, 255))
        background.paste(image, mask=image.split()[3])
        return background
    return image.convert("RGB")


_DCT = None


def _dct_matrix(n=32):
    global _DCT
    if _DCT is None:
        k = np.arange(n)[:, None]
        i = np.arange(n)[None, :]
        m = np.cos(np.pi * (2 * i + 1) * k / (2 * n)) * math.sqrt(2 / n)
        m[0] /= math.sqrt(2)
        _DCT = m
    return _DCT


def features(image):
    """pHash (64-bit), dHash (64-bit), 32x32 grayscale thumbnail and a decoded-pixel hash."""
    rgb = to_rgb(image)
    gray = rgb.convert("L")
    small = np.asarray(gray.resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    d = _dct_matrix()
    coeffs = (d @ small @ d.T)[:8, :8].flatten()
    phash = _bits(coeffs > np.median(coeffs[1:]))
    diff = np.asarray(gray.resize((9, 8), Image.Resampling.LANCZOS), dtype=np.int16)
    dhash = _bits((diff[:, 1:] > diff[:, :-1]).flatten())
    thumb = np.asarray(gray.resize((THUMB, THUMB), Image.Resampling.BILINEAR), dtype=np.uint8).tobytes()
    pixel_sha = hashlib.sha256(f"{rgb.size}".encode() + rgb.tobytes()).hexdigest()
    return {"phash": phash, "dhash": dhash, "thumb": thumb, "pixel_sha256": pixel_sha}


def _bits(bools):
    value = 0
    for b in bools:
        value = (value << 1) | int(b)
    return value - (1 << 64) if value >= 1 << 63 else value


def thumb_similarity(a, b):
    """Pearson correlation of two thumbnails; 1.0 for identical, constant images compare by mean."""
    x = np.frombuffer(a, dtype=np.uint8).astype(np.float64)
    y = np.frombuffer(b, dtype=np.uint8).astype(np.float64)
    sx, sy = x.std(), y.std()
    if sx < 1 or sy < 1:
        return 1.0 - abs(x.mean() - y.mean()) / 255.0 if sx < 1 and sy < 1 else 0.0
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def store_path(sha, ext):
    return Path("images") / sha[:2] / f"{sha}.{ext}"


def write_asset(root, data, ext):
    """Write bytes to the content-addressed store once; return (sha256, relative path)."""
    sha = hashlib.sha256(data).hexdigest()
    rel = store_path(sha, ext)
    path = Path(root) / rel
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)
    return sha, rel


def prepare(data, root=None):
    """Decode, validate and (optionally) store an image; return its record fields."""
    image, fmt = decode(data)
    transform = None
    if has_unhandled_alpha(image):
        original = hashlib.sha256(data).hexdigest()
        data = normalize_alpha(image)
        image, fmt = decode(data)
        transform = {"alpha_composite_white": original}
    width, height = image.size
    if max(width, height) / min(width, height) > QWEN_MAX_RATIO:
        raise ImageError("image_aspect_ratio")
    ext = FORMATS[fmt]
    if root is not None:
        sha, rel = write_asset(root, data, ext)
    else:
        sha, rel = hashlib.sha256(data).hexdigest(), store_path(hashlib.sha256(data).hexdigest(), ext)
    return {"image_sha256": sha, "image_path": str(rel), "width": width, "height": height, "image_format": ext, "image_mode": image.mode, "image_bytes": len(data), "image_transform": transform, **features(image)}


def qwen_resize(width, height):
    """qwen_vl_utils.smart_resize as called by fetch_image for a path input; returns (width, height)."""
    f = QWEN_FACTOR

    def rnd(x):
        return round(x / f) * f

    h_bar, w_bar = max(f, rnd(height)), max(f, rnd(width))
    if h_bar * w_bar > QWEN_MAX_PIXELS:
        beta = math.sqrt(height * width / QWEN_MAX_PIXELS)
        h_bar, w_bar = math.floor(height / beta / f) * f, math.floor(width / beta / f) * f
    elif h_bar * w_bar < QWEN_MIN_PIXELS:
        beta = math.sqrt(QWEN_MIN_PIXELS / (height * width))
        h_bar, w_bar = math.ceil(height * beta / f) * f, math.ceil(width * beta / f) * f
    return w_bar, h_bar


def image_tokens(width, height, processor_config):
    """Image-pad tokens the K2-Horizon-V processor emits after Miles's qwen pre-resize."""
    from transformers.models.glm5_next.image_processing_pil_glm5_next import smart_resize

    w, h = qwen_resize(width, height)
    factor = processor_config["patch_size"] * processor_config["merge_size"]
    th, tw = smart_resize(num_frames=processor_config["temporal_patch_size"], height=h, width=w, temporal_factor=processor_config["temporal_patch_size"], factor=factor, min_pixels=processor_config["min_image_tokens"], max_pixels=processor_config["max_image_tokens"])
    return (th // factor) * (tw // factor), (tw, th)


def unpatchify(pixel_values, grid_thw, image_processor):
    """Invert GLM patchify and normalization: the exact pixels the vision tower receives."""
    p, m = image_processor.patch_size, image_processor.merge_size
    t, gh, gw = (int(x) for x in grid_thw)
    c, tp = 3, image_processor.temporal_patch_size
    x = np.asarray(pixel_values, dtype=np.float32).reshape(gh // m, gw // m, m, m, c, tp, p, p)[:, :, :, :, :, 0]
    x = x.transpose(4, 0, 2, 5, 1, 3, 6).reshape(c, gh * p, gw * p)
    mean = np.asarray(image_processor.image_mean, dtype=np.float32)[:, None, None]
    std = np.asarray(image_processor.image_std, dtype=np.float32)[:, None, None]
    x = np.clip((x * std + mean) * 255.0, 0, 255).round().astype(np.uint8)
    return Image.fromarray(x.transpose(1, 2, 0))


def fit(image, size):
    return ImageOps.contain(image, (size, size))
