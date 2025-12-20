import instructor
from litellm import completion
from langsmith import traceable, get_current_run_tree
from jinja2 import Template
from src.backend.app.models.schemas import StylistAgentResponse, FashionSet
from src.backend.app.config import settings
from src.backend.app.dependencies import deps


def format_item_list(item_list: dict[str, str]) -> str:
    """Format the item list for the stylist agent."""
    if item_list is None:
        return ""
    return "\n".join([f"{item}: {description}" for item, description in item_list.items()])


@traceable(
    name="stylist_agent",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": settings.llm_model
    },
)
def stylist_agent(user_intention: str, item_list: str = None):
    prompt_template = deps.prompt_manager.get_prompt("recommender")

    template = Template(prompt_template)
    prompt = template.render(user_intention=user_intention, item_list=item_list)

    client = instructor.from_litellm(completion)

    response, raw_response = client.chat.completions.create_with_completion(
        model=settings.llm_model,
        messages=[{
            "role": "user",
            "content": prompt
        }],
        response_model=StylistAgentResponse,
        temperature=0.5,
    )

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    return response.recommendations


def parse_recommendations(recommendations: dict[str, FashionSet]) -> str:
    if isinstance(recommendations, dict):
        items_to_parse = recommendations.items()
    else:
        raise ValueError("recommendations must be a dictionary")

    output_parts = []
    for fashion_set_name, fashion_set in recommendations.items():
        output = f"{fashion_set_name}:\n"
        output += f"\tItems: {fashion_set.items}\n"
        output += f"\tReason: {fashion_set.reason}\n"
        output_parts.append(output)
    return "\n".join(output_parts)


def get_recommendations(user_intention: str, item_list: dict[str, str] = None) -> str:
    """Get recommendations for a given query and item list from a fashion expert.

    Args:
        user_intention: The user's intention.
        item_list: Dictionary with item name as the key and item description as the value.

    Returns:
        A string of the recommendations.
    """

    item_list = format_item_list(item_list)
    recommendations = stylist_agent(user_intention, item_list)
    parsed_recommendations = parse_recommendations(recommendations)
    return parsed_recommendations
