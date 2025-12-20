from urllib.parse import urlparse
import os
import base64
from PIL import Image
from io import BytesIO
import requests


def get_image_from_source(path, bbox=None) -> Image.Image:
    url_scheme = urlparse(path)
    if url_scheme.scheme in ("http", "https"):
        response = requests.get(path)
        image = Image.open(BytesIO(response.content))
    elif os.path.exists(path):
        image = Image.open(path)
    else:
        raise ValueError(f"Invalid image source: {path}")
    if bbox is not None:
        return image.crop(bbox)
    return image


def pil_to_base64_png(img: Image.Image):
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")
