from pydantic import BaseModel, Field, field_validator
from typing import Optional, Tuple, Literal, Annotated, List, Dict, Any
from urllib.parse import urlparse
from operator import add
import os
from langchain_core.messages import AIMessage, ToolMessage


# Image
class ImageSource(BaseModel):
    """Reference to an image with optional bounding box for cropping."""
    path: str = Field(..., description="URL (http/https) or local file path to the image.")
    bbox: Optional[Tuple[float, float, float, float]] = Field(
        default=None,
        description="Bounding box as (left, upper, right, lower) for cropping.",
    )

    @field_validator("path")
    @classmethod
    def validate_path(cls, v: str) -> str:
        if v.startswith("data:image"):
            return v

        url_scheme = urlparse(v)
        if url_scheme.scheme in ("http", "https"):
            return v
        elif os.path.exists(v):
            return v
        raise ValueError(f"Path must be a valid URL or local file path: {v}")


# Descriptor
class ItemDescription(BaseModel):
    item_name: str = Field(..., description="Name of the item.")
    item_description: str = Field(..., description="Description of the item.")


class DescriptorAgentResponse(BaseModel):
    item_descriptions: dict[str, list[ItemDescription]] = Field(
        ..., description="A dictionary with image id as the key and a dictionary of item name and description as the value.")


# Recommender
class FashionSet(BaseModel):
    items: list[str] = Field(..., description="List of fashion items in this set")
    reason: str = Field(..., description="Explanation of why this set works together")


class StylistAgentResponse(BaseModel):
    recommendations: dict[str, FashionSet] = Field(
        ...,
        description="A dictionary with fashion set name as the key and its details as the value.",
    )


# Agent
class ToolCall(BaseModel):
    name: str
    arguments: dict


class RetrievedImages(BaseModel):
    type: Literal["retrieved"] = Field("retrieved", description="Type discriminator for retrieved images.")
    image_ids: list[str] = Field(default_factory=list, description="List of image ids retrieved from the database.")


class UserProvidedImages(BaseModel):
    type: Literal["user_provided"] = Field("user_provided", description="Type discriminator for user provided images.")
    image_ids: list[str] = Field(default_factory=list, description="List of image ids used to get descriptions.")


class AgentResponse(BaseModel):
    answer: str = Field(description="Answer to the question.")
    final_answer: bool = False
    tool_calls: List[ToolCall] = Field(default_factory=list)
    image_ids: RetrievedImages | None = None


# Graph State
class State(BaseModel):
    messages: Annotated[List[Any], add] = Field(default_factory=list)
    session_id: Optional[str] = Field(None, description="The session id to use for the chat.")
    iteration: int = 0
    answer: str = ""
    available_tools: List[Dict[str, Any]] = Field(default_factory=list)
    tool_calls: List[ToolCall] = Field(default_factory=list)
    final_answer: bool = False
    image_ids: list[RetrievedImages | UserProvidedImages] = Field(default_factory=list)


# Frontend
class ChatRequest(BaseModel):
    query: str = Field(..., description="The query to send to the chatbot.")
    session_id: Optional[str] = Field(None, description="The session id to use for the chat.")
    images: Optional[List[str]] = Field(None, description="The list of image urls or local file paths to the images.")


class ImageResult(BaseModel):
    image_id: str
    url: str
    bbox: Optional[Tuple[float, float, float, float]] = None
    type: Literal["user_provided", "retrieved"]


class ChatResponse(BaseModel):
    answer: str = Field(..., description="The answer to the query.")
    session_id: str = Field(..., description="Session ID to use for subsequent requests.")
    images: Optional[List[ImageResult]] = Field(None, description="Images referenced in the response.")


# Session
class Session(BaseModel):
    image_source_store: dict[str, ImageSource] = Field(default_factory=dict)
    image_ids_store: list[UserProvidedImages | RetrievedImages] = Field(default_factory=list)
    message_history: list[AIMessage | ToolMessage | dict[str, Any]] = Field(default_factory=list)
