from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import AIMessage
from src.backend.app.models.schemas import State
from src.backend.app.dependencies import deps
from src.backend.app.services.descriptor import create_get_item_descriptions
from src.backend.app.services.recommender import get_recommendations
from src.backend.app.services.retrieval import create_retrieve_item_from_wardrobe
from src.backend.app.services.search import search_item
from src.backend.app.services.vton import create_virtual_try_on_image
from src.backend.app.utils.utils import get_tool_descriptions, add_image_ids_to_message, load_message_history_for_llm
from src.backend.app.services.agent import agent_node
from src.backend.app.models.schemas import ChatRequest, ChatResponse, ImageResult, ImageSource
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor


def tool_router(state: State) -> str:
    """Decide whether to call a tool or end"""

    if state.final_answer:
        return "end"
    elif state.iteration > 5:
        return "end"
    elif len(state.tool_calls) > 0:
        return "tools"
    else:
        return "end"


def get_graph(
    session_id: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    q_client: QdrantClient,
) -> tuple[StateGraph, list[str]]:
    workflow = StateGraph(State)

    descriptor_tool = [
        create_get_item_descriptions(session_id),
        get_recommendations,
        create_retrieve_item_from_wardrobe(session_id, model, processor, q_client),
        search_item,
        create_virtual_try_on_image(session_id),
    ]
    tool_node = ToolNode(descriptor_tool)
    tool_descriptions = get_tool_descriptions(descriptor_tool)

    workflow.add_node("agent_node", agent_node)
    workflow.add_node("tool_node", tool_node)

    workflow.add_edge(START, "agent_node")
    workflow.add_conditional_edges(
        "agent_node",
        tool_router,
        {
            "tools": "tool_node",
            "end": END,
        },
    )
    workflow.add_edge("tool_node", "agent_node")

    graph = workflow.compile()

    return graph, tool_descriptions


def invoke_graph(
    chat_request: ChatRequest,
    qdrant_client: QdrantClient,
    clip: tuple,
) -> dict:

    model, processor = clip
    session_id = deps.session_manager.get_or_create_session(chat_request.session_id)

    # load message history and previous image_ids
    message_history = deps.session_manager.load_message_history(session_id)
    updated_message_history = []
    for message in message_history:
        edited_message = load_message_history_for_llm(message)
        updated_message_history.append(edited_message)

    # converting image urls or local file paths to image ids
    user_provided_image_ids = []
    user_provided_images = []
    if chat_request.images:
        for i, image in enumerate(chat_request.images):
            image_source = ImageSource(path=image, bbox=None)
            user_provided_image_id = deps.session_manager.store_image_source(session_id, image_source)
            user_provided_image_ids.append(user_provided_image_id)
            user_provided_images.append(ImageResult(
                image_id=user_provided_image_id,
                url=image,
                bbox=None,
                type="user_provided",
            ))

    if chat_request.model_image:
        model_image_source = ImageSource(path=chat_request.model_image, bbox=None)
        model_image_id = deps.session_manager.store_image_source(session_id, model_image_source, is_model=True)

    graph, tool_descriptions = get_graph(session_id, model, processor, qdrant_client)

    # appending the image_ids to the query
    user_query = add_image_ids_to_message(chat_request.query, user_provided_image_ids, type="user_provided")
    query = {"role": "user", "content": user_query}

    # initializing the state
    initial_state = {
        "messages": updated_message_history + [query],
        "available_tools": tool_descriptions,
        "session_id": session_id,
    }

    result = graph.invoke(initial_state)

    ai_result_images = result.get("images", [])
    ai_images = []
    for ai_result_image in ai_result_images:
        img_source = deps.session_manager.get_image_source(session_id, ai_result_image.image_id)
        ai_images.append(ImageResult(
            image_id=ai_result_image.image_id,
            url=img_source.path,
            bbox=img_source.bbox,
            type=ai_result_image.type,
        ))

    # store the original messages
    deps.session_manager.store_message(
        session_id,
        user_query=chat_request.query,
        ai_response=result.get("answer", ""),
        user_images=user_provided_images if user_provided_images else None,
        ai_images=ai_images if ai_images else None,
    )

    return ChatResponse(
        session_id=session_id,
        answer=result.get("answer", ""),
        images=ai_images if ai_images else None,
    )
