from typing import List
from torch import torch
from langsmith import traceable
from src.backend.app.models.schemas import ImageSource
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor
from src.backend.app.config import settings
from src.backend.app.dependencies import deps


@traceable(
    name="get_text_features",
    run_type="embedding",
    metadata={
        "ls_provider": "huggingface",
        "ls_model_name": settings.clip_model_name
    },
)
def get_text_features(model: CLIPModel, processor: CLIPProcessor, text_query: list[str]):
    text_inputs = processor.tokenizer(text=text_query, return_tensors="pt", padding=True)
    input_ids = text_inputs["input_ids"].cuda()
    attention_mask = text_inputs["attention_mask"].cuda()
    with torch.no_grad():
        text_features = model.get_text_features(input_ids, attention_mask)
    return text_features.cpu().numpy().tolist()


@traceable(
    name="retrieve_item",
    run_type="retriever",
)
def retrieve_item(text_features: List[float], q_client: QdrantClient, top_k: int = 1):
    payloads = []
    for text_feature in text_features:
        payloads.append(
            q_client.query_points(collection_name="ctl_dataset_train_sample_500", query=text_feature, with_payload=True,
                                  limit=top_k).points[0].payload)

    images = [ImageSource(path=item["image_url"], bbox=item["bbox"]) for item in payloads]

    return images


def parse_retrieved_items(image_ids: list[str], item_list: list[str]) -> str:
    """Parse the retrieved items from the database.

    Args:
        image_ids: List of image ids.
        item_list: List of item names.
    
    Returns:
        A string of the retrieved items.
    """
    return "\n".join([f"{image_id}: {item}" for item, image_id in zip(item_list, image_ids)])


def create_retrieve_item_from_wardrobe(
    session_id: str,
    model: CLIPModel,
    processor: CLIPProcessor,
    q_client: QdrantClient,
):

    def retrieve_item_from_wardrobe(item_list: list[str]) -> str:
        """Retrieve items from the wardrobe.

        Args:
            item_list: List of item names.

        Returns:
            A dictionary with item name as the key and image id as the value.
        """
        text_features = get_text_features(model, processor, item_list)
        images = retrieve_item(text_features, q_client, top_k=1)
        image_ids = [deps.session_manager.store_image_source(session_id, image) for image in images]
        parsed_retrieved_items = parse_retrieved_items(image_ids, item_list)
        return parsed_retrieved_items

    return retrieve_item_from_wardrobe
