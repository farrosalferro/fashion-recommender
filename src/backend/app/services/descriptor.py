import instructor
from litellm import completion
from langsmith import traceable, get_current_run_tree
from instructor.processing import multimodal
from jinja2 import Template
from src.backend.app.models.schemas import DescriptorAgentResponse, ItemDescription
from src.backend.app.config import settings
from src.backend.app.dependencies import deps


def format_image_list(image_id_list: list[str], session_id: str) -> dict[str, str]:
    """Format the image list for the descriptor agent."""
    return {image_id: deps.session_manager.get_image_source(session_id, image_id) for image_id in image_id_list}


def process_images(image_url: str):
    image = multimodal.Image.from_url(image_url)
    return image


@traceable(
    name="descriptor_agent",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": settings.llm_model
    },
)
def descriptor_agent(image_list: dict[str, str]):
    prompt_template = deps.prompt_manager.get_prompt("descriptor")

    template = Template(prompt_template)

    prompt = [template.render()]

    for image_id, image_info in image_list.items():
        prompt.append(image_id)
        prompt.append(process_images(image_info.path))

    client = instructor.from_litellm(completion)

    response, raw_response = client.chat.completions.create_with_completion(
        model=settings.llm_model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        response_model=DescriptorAgentResponse,
        temperature=0.5,
    )

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    return response.item_descriptions


def parse_item_descriptions(item_descriptions: dict[str, ItemDescription]) -> str:
    """Format the item descriptions from the descriptor agent."""

    if isinstance(item_descriptions, dict):
        items_to_parse = item_descriptions.items()
    else:
        raise ValueError("item_descriptions must be a dictionary")

    output_parts = []
    for image_id, item_descriptions in items_to_parse:
        output = f"{image_id}:\n"

        for item_description in item_descriptions:
            output += f"\t{item_description.item_name}: {item_description.item_description}\n"
        output_parts.append(output)
    return "\n".join(output_parts)


def create_get_item_descriptions(session_id: str):

    def get_item_descriptions(image_id_list: list[str]) -> str:
        """Get descriptions for a list of images.

        Args:
            image_id_list: The list of image ids to get descriptions from.

        Returns:
            A string of the item descriptions.
        """

        image_dict = format_image_list(image_id_list, session_id)
        item_descriptions = descriptor_agent(image_dict)
        formatted_item_descriptions = parse_item_descriptions(item_descriptions)
        return formatted_item_descriptions

    return get_item_descriptions
