from langsmith import traceable, get_current_run_tree
from PIL import Image
from jinja2 import Template
from google import genai
from src.backend.app.config import settings
from src.backend.app.dependencies import deps
from src.backend.app.models.schemas import ImageSource
from src.backend.app.utils.image_utils import bytes_to_base64_data_url, get_image_from_source


@traceable(
    name="vton_agent",
    run_type="llm",
    metadata={
        "ls_provider": "google",
        "ls_model_name": settings.vton_model,
    },
)
def virtual_try_on_agent(model_image: Image.Image, item_images: list[Image.Image]) -> tuple[bytes, str]:

    prompt_template = deps.prompt_manager.get_prompt("vton")

    template = Template(prompt_template)

    prompt = template.render()

    g_client = genai.Client()
    response = g_client.models.generate_content(
        model=settings.vton_model,
        contents=[model_image, *item_images, prompt],
    )

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": response.usage_metadata.prompt_token_count,
            "output_tokens": response.usage_metadata.candidates_token_count,
            "total_tokens": response.usage_metadata.total_token_count
        }

    image_parts = response.parts[0].as_image()
    image_bytes, mime_type = image_parts.image_bytes, image_parts.mime_type

    return image_bytes, mime_type


def create_virtual_try_on_image(session_id: str):

    def get_virtual_try_on_image(item_image_ids: list[str]) -> str:
        """Get a virtual try-on image of a model wearing a new outfit.

        Args:
            item_image_ids: The image ids of the item images.

        Returns:
            An image id of the virtual try-on image.
        """

        model_image_source = deps.session_manager.get_model_source(session_id)
        if model_image_source is None:
            return "[ERROR] User has not uploaded his / her photo yet."

        item_image_sources = [deps.session_manager.get_image_source(session_id, item_image_id) for item_image_id in item_image_ids]

        model_image = get_image_from_source(model_image_source.path, model_image_source.bbox)
        item_images = [get_image_from_source(item_image_source.path, item_image_source.bbox) for item_image_source in item_image_sources]

        image_bytes, mime_type = virtual_try_on_agent(model_image, item_images)
        virtual_try_on_image = bytes_to_base64_data_url(image_bytes, mime_type)
        virtual_try_on_image_id = deps.session_manager.store_image_source(
            session_id,
            ImageSource(path=virtual_try_on_image, bbox=None),
        )

        return virtual_try_on_image_id

    return get_virtual_try_on_image
