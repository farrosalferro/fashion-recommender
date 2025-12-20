from langsmith import traceable
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

    prompt_template = deps.prompt_manager.get_prompt("agent")

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
    if response.image_ids:
        # do not insert if the image_ids is the same as the last one
        if len(response.image_ids.image_ids) > 0:
            if len(state.image_ids) > 0 and response.image_ids.image_ids == state.image_ids[-1].image_ids:
                pass
            else:
                state.image_ids.append(response.image_ids)

    return {
        "messages": [ai_message],
        "tool_calls": response.tool_calls,
        "iteration": state.iteration + 1,
        "answer": response.answer,
        "final_answer": response.final_answer,
        "image_ids": state.image_ids
    }
