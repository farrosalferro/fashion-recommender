from langsmith import traceable, get_current_run_tree
from src.backend.app.models.schemas import AgentResponse, State
from jinja2 import Template
import instructor
from litellm import completion
from langchain_core.messages import convert_to_openai_messages
from src.backend.app.utils.utils import format_ai_message
from src.backend.app.config import settings
from src.backend.app.dependencies import deps


@traceable(
    name="agent_node",
    run_type="llm",
    metadata={
        "ls_provider": "openai",
        "ls_model_name": settings.llm_model
    },
)
def agent_node(state: State) -> dict:

    # prompt_template = deps.prompt_manager.get_prompt("agent")
    # FOR DEBUGGING ONLY
    import yaml
    agent_prompt_path = "/home/farrosalferro/projects/fashion_recommender/prompts/agent.yaml"
    with open(agent_prompt_path, "r") as file:
        prompt_template = yaml.safe_load(file)["prompt"]

    template = Template(prompt_template)

    prompt = template.render(available_tools=state.available_tools)

    conversations = []
    messages = state.messages

    for message in messages:
        conversations.append(convert_to_openai_messages(message))

    client = instructor.from_litellm(completion)
    response, raw_response = client.chat.completions.create_with_completion(
        model=settings.llm_model,
        response_model=AgentResponse,
        messages=[{
            "role": "system",
            "content": prompt
        }, *conversations],
    )

    ai_message = format_ai_message(response)
    for image in (response.images or []):
        state.images.append(image)

    current_run = get_current_run_tree()

    if current_run:
        current_run.metadata["usage_metadata"] = {
            "input_tokens": raw_response.usage.prompt_tokens,
            "output_tokens": raw_response.usage.completion_tokens,
            "total_tokens": raw_response.usage.total_tokens
        }

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "images": state.images
    }
