from urllib.parse import urlparse
from typing import Optional, Tuple
import os
import base64
from PIL import Image
from io import BytesIO
import requests


def get_image_from_source(path: str, bbox: Optional[Tuple[float, float, float, float]] = None) -> Image.Image:
    if path.startswith("data:image"):
        encoded = path.split(",")[1]
        image_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_bytes))
    elif urlparse(path).scheme in ("http", "https"):
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


def bytes_to_base64_data_url(image_bytes: bytes, mime_type: str) -> str:
    return f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('utf-8')}"
