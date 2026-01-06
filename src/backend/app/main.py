from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.backend.app.dependencies import deps
from src.backend.app.config import settings
from qdrant_client import QdrantClient
from transformers import CLIPModel, CLIPProcessor
from src.backend.app.services.session import SessionManager
from src.backend.app.models.schemas import ChatRequest, SessionDataResponse
from src.backend.app.services.graph import invoke_graph
from src.backend.app.prompt_manager import PromptManager


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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    session_data = deps.session_manager.get_session_data(session_id)
    if session_data is None:
        raise HTTPException(status_code=404, detail="Session not found")

    return session_data
