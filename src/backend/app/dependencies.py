from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor
from src.backend.app.services.session import SessionManager
from src.backend.app.prompts import PromptManager


class AppDependencies:
    """Container for application-wide dependencies."""
    qdrant_client: QdrantClient | None = None
    clip_model: CLIPModel | None = None
    clip_processor: CLIPProcessor | None = None
    session_manager: SessionManager | None = None
    prompt_manager: PromptManager | None = None


deps = AppDependencies()
