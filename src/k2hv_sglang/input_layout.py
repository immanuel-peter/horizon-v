import hashlib


def image_hash(pixel_values, image_grid_thw):
    digest = hashlib.sha256()
    for tensor in (image_grid_thw, pixel_values):
        digest.update(memoryview(tensor.detach().contiguous().numpy()))
    return int.from_bytes(digest.digest()[:8], "big")


def expand_image_tokens(input_ids, image_grid_thw, *, image_start_id, image_token_id, image_end_id, merge_size=2):
    """Expand single-pad markers or validate expanded markers without changing text IDs."""
    counts = [int(grid.prod()) // merge_size**2 for grid in image_grid_thw]
    result, offsets, image_index = [], [], 0
    index = 0
    while index < len(input_ids):
        token = input_ids[index]
        if token == image_start_id:
            end = index + 1
            while end < len(input_ids) and input_ids[end] == image_token_id:
                end += 1
            if end == index + 1 or end == len(input_ids) or input_ids[end] != image_end_id:
                raise ValueError("Each image must have a complete start/pad/end token block")
            if image_index >= len(counts):
                raise ValueError("More image token blocks than supplied images")
            expected = counts[image_index]
            if end - index - 1 not in (1, expected):
                raise ValueError(f"Image {image_index} has {end - index - 1} placeholder tokens, expected 1 or {expected}")
            result.append(image_start_id)
            start = len(result)
            result.extend([image_token_id] * expected)
            offsets.append((start, len(result) - 1))
            result.append(image_end_id)
            image_index += 1
            index = end + 1
        elif token in (image_token_id, image_end_id):
            raise ValueError("Image tokens must occur inside complete start/pad/end blocks")
        else:
            result.append(token)
            index += 1
    if image_index != len(counts):
        raise ValueError("More supplied images than image token blocks")
    return result, offsets


def split_image_features(pixel_values, image_grid_thw, *, merge_size=2):
    if image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3:
        raise ValueError("image_grid_thw must have shape (num_images, 3)")
    if image_grid_thw.numel() and (bool((image_grid_thw <= 0).any()) or bool((image_grid_thw[:, 1:] % merge_size).any())):
        raise ValueError("Image grids must be positive with spatial dimensions divisible by merge_size")
    lengths = image_grid_thw.prod(-1).tolist()
    if pixel_values.ndim != 2 or sum(lengths) != pixel_values.shape[0]:
        raise ValueError("Image grid patch counts do not match pixel_values")
    return list(zip(pixel_values.split(lengths), image_grid_thw.split(1)))


def map_vision_weight(name):
    for source, shard in ((".mlp.gate_proj.", 0), (".mlp.up_proj.", 1)):
        if source in name:
            return name.replace(source, ".mlp.gate_up_proj."), shard
    return name.replace(".attn.qkv.", ".attn.qkv_proj."), None
