from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel
import torch
from io import BytesIO
from itertools import batched
import pandas as pd
import argparse
from typing import Optional, Tuple
import os

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client import models

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_image_features_from_df(
    model: CLIPModel,
    processor: CLIPProcessor,
    df: pd.DataFrame,
) -> list[float]:

    unique_image_signatures = df["image_signature"].unique()
    image_features = []
    for image_signature in unique_image_signatures:
        img_url = df[df["image_signature"] == image_signature]["image_url"].values[0]
        bboxes = df[df["image_signature"] == image_signature]["bbox"]
        cropped_images = [get_single_image_from_url(img_url, bbox) for bbox in bboxes]
        image_features.extend(get_image_features(model, processor, cropped_images))
    return image_features


def get_image_features(
    model: CLIPModel,
    processor: CLIPProcessor,
    images: list[Image.Image],
) -> list[float]:

    pixel_values = processor.image_processor.preprocess(images, return_tensors="pt")['pixel_values'].to(device)
    with torch.no_grad():
        image_features = model.get_image_features(pixel_values)
    return image_features.cpu().numpy().tolist()


def get_single_image_from_url(
    url: str,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> Image.Image:

    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    if bbox is not None:
        return image.crop(bbox)
    return image


def create_collection(
    client: QdrantClient,
    collection_name: str,
    data_to_embed: list[dict],
    embedding_length: int = 512,
    batch_size: int = 100,
) -> None:

    try:
        client.create_collection(collection_name=collection_name, vectors_config=VectorParams(
            size=embedding_length,
            distance=Distance.COSINE,
        ))

        for batch in batched(data_to_embed, batch_size):
            ids = [item["id"] for item in batch]
            vectors = [item["vector"] for item in batch]
            payloads = [item["payload"] for item in batch]
            client.upsert(
                collection_name=collection_name,
                points=models.Batch(
                    ids=ids,
                    vectors=vectors,
                    payloads=payloads,
                ),
            )

        print(f"Collection {collection_name} created successfully")

    except Exception as e:
        raise Exception(f"Error creating collection: {e}")


def main():
    parser = argparse.ArgumentParser(description="Create a collection in Qdrant")
    parser.add_argument("--data_path", default="./data/example/sample_5.jsonl", type=str, help="The path to the data file")
    parser.add_argument("--collection_name", default="ctl_sample_5", type=str, help="The name of the collection")
    parser.add_argument("--qdrant_url", type=str, default="http://localhost:6333", help="The URL of the Qdrant server")
    parser.add_argument("--clip_model_name", type=str, default="patrickjohncyh/fashion-clip", help="The name of the CLIP model")
    parser.add_argument("--batch_size", type=int, default=100, help="The batch size for the upsert")
    args = parser.parse_args()

    # initialize client and model
    client = QdrantClient(url=args.qdrant_url)
    model = CLIPModel.from_pretrained(args.clip_model_name).to(device)
    processor = CLIPProcessor.from_pretrained(args.clip_model_name)

    # check if collection exists
    if client.collection_exists(args.collection_name):
        raise ValueError(f"Collection {args.collection_name} already exists")

    # check if data_path exists and is .jsonl file
    if not os.path.exists(args.data_path):
        raise FileNotFoundError(f"Data file {args.data_path} not found")
    if not args.data_path.endswith(".jsonl"):
        raise ValueError(f"Data file {args.data_path} is not a .jsonl file")

    df = pd.read_json(args.data_path, orient="records", lines=True)
    image_features = get_image_features_from_df(model, processor, df)
    embedding_length = len(image_features[0])

    image_signatures = df["image_signature"].values
    labels = df["label"].values
    img_urls = df["image_url"].values
    ids = [i for i in range(len(image_signatures))]

    bboxes = df["bbox"].values
    data_to_embed = []
    for i, image_signature, label, image_feature, img_url, bbox in zip(ids, image_signatures, labels, image_features, img_urls, bboxes):
        data_to_embed.append({
            "id": i,
            "vector": image_feature,
            "payload": {
                "image_signature": image_signature,
                "label": label,
                "image_url": img_url,
                "bbox": bbox
            }
        })

    create_collection(
        client=client,
        collection_name=args.collection_name,
        data_to_embed=data_to_embed,
        embedding_length=embedding_length,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
