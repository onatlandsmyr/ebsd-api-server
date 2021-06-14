import io
import numpy as np
from PIL import Image
import base64
import skimage
from typing import Literal


def get_image_from_array(
    data,
    encoding: Literal["base64", "binary"] = "base64",
    percentiles=None,
    rescale=False,
    mode=None,
    format="PNG",
):
    if percentiles is not None or rescale:
        data = rescale_intensities(data, percentiles)
    im = Image.fromarray(data, mode=mode)
    image = io.BytesIO()
    im.save(image, format=format)
    image.seek(0)
    if encoding is "binary":
        return image
    return f"data:image/png;base64,{base64.b64encode(image.getvalue()).decode('ascii')}"


def rescale_intensities(data, percentiles=None):
    if percentiles is not None:
        in_range = tuple(np.percentile(data, q=percentiles))
    else:
        in_range = "image"
    data = skimage.exposure.rescale_intensity(
        data, in_range=in_range, out_range=(0, 255)
    ).astype(np.uint8)
    return data
