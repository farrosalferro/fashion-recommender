from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from src.backend.app.dependencies import deps
from src.backend.app.config import settings
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor
from src.backend.app.services.session import SessionManager
from src.backend.app.models.schemas import ChatRequest
from src.backend.app.services.graph import invoke_graph
from src.backend.app.prompts import PromptManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    deps.qdrant_client = QdrantClient(url=settings.qdrant_url)
    deps.clip_model = CLIPModel.from_pretrained(settings.clip_model_name).cuda()
    deps.clip_processor = CLIPProcessor.from_pretrained(settings.clip_model_name)
    deps.session_manager = SessionManager()
    deps.prompt_manager = PromptManager()

    yield

    deps.qdrant_client.close()


app = FastAPI(lifespan=lifespan)


def get_qdrant_client() -> QdrantClient:
    return deps.qdrant_client


def get_clip_model():
    return deps.clip_model, deps.clip_processor


@app.post("/chat")
async def chat(
        request: ChatRequest,
        qdrant_client: QdrantClient = Depends(get_qdrant_client),
        clip: tuple = Depends(get_clip_model),
):
    return invoke_graph(request, qdrant_client, clip)
