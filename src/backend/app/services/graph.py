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
from src.backend.app.utils.utils import get_tool_descriptions
from src.backend.app.services.agent import agent_node
from src.backend.app.models.schemas import ChatRequest, ChatResponse, ImageResult, UserProvidedImages, ImageSource, RetrievedImages
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

    # TODO: remove this part as the ids are automatically added to the history
    # image_ids_history = deps.session_manager.get_image_ids(session_id)
    # # check if the last message retrieved images
    # if (len(message_history) > 0 and isinstance(message_history[-1], AIMessage)) \
    #     and (len(image_ids_history) > 0 and isinstance(image_ids_history[-1], RetrievedImages)):

    #     # converting retrieved images to text
    #     retrieved_image_ids_text = "\nRetrieved images:"
    #     last_image_ids = image_ids_history[-1].image_ids
    #     for i, last_image_id in enumerate(last_image_ids):
    #         retrieved_image_ids_text += f"\n{i+1}. {last_image_id}"
    #     last_message = message_history.pop().content + retrieved_image_ids_text
    #     message_history.append(AIMessage(content=last_message))

    # converting image urls or local file paths to image ids
    user_provided_image_ids_text = ""
    user_provided_images = None
    image_ids = []
    if chat_request.images:
        user_provided_image_ids_text = "\nUser provided fashion item images:"
        for i, image in enumerate(chat_request.images):
            image_source = ImageSource(path=image, bbox=None)
            image_id = deps.session_manager.store_image_source(session_id, image_source)
            image_ids.append(image_id)
            user_provided_image_ids_text += f"\n{i+1}. {image_id}"
        user_provided_images = UserProvidedImages(image_ids=image_ids)

    if chat_request.model_image:
        model_image_source = ImageSource(path=chat_request.model_image, bbox=None)
        model_image_id = deps.session_manager.store_image_source(session_id, model_image_source, is_model=True)

    graph, tool_descriptions = get_graph(session_id, model, processor, qdrant_client)

    # appending the image_ids to the query
    query = {"role": "user", "content": chat_request.query + user_provided_image_ids_text}

    # initializing the state
    initial_state = {
        "messages": message_history + [query],
        "available_tools": tool_descriptions,
        "session_id": session_id,
        "image_ids": [],
    }

    result = graph.invoke(initial_state)

    user_images = []
    if user_provided_images:
        for user_img_id in user_provided_images.image_ids:
            img_source = deps.session_manager.get_image_source(session_id, user_img_id)
            user_images.append(ImageResult(
                image_id,
                url=img_source.path,
                bbox=img_source.bbox,
                type="user_provided",
            ))

    # store the messages

    ai_images = []
    ai_image_ids_text = ""
    for img_group in result.get("image_ids", []):
        deps.session_manager.store_image_ids(session_id, img_group)  # TODO: fix store and get image ids since images are attached to conversations
        if img_group.type == "retrieved":
            ai_image_ids_text += "Retrieved image ids:\n"
        else:
            ai_image_ids_text += "Virtual try-on image ids: \n"
        for img_id in img_group.image_ids:
            img_source = deps.session_manager.get_image_source(session_id, img_id)
            ai_images.append(ImageResult(
                image_id=img_id,
                url=img_source.path,
                bbox=img_source.bbox,
                type=img_group.type,
            ))
            ai_image_ids_text += f"\t{img_id}\n"

    deps.session_manager.store_message(
        session_id,
        user_query=query.get("content"),
        ai_message=result["messages"][-1].content + ai_image_ids_text,
        user_images=user_images if user_images else None,
        ai_images=ai_images if ai_images else None,
    )

    return ChatResponse(
        session_id=session_id,
        answer=result.get("answer", ""),
        images=ai_images if ai_images else None,
    )
