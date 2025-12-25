from src.backend.app.models.schemas import ImageSource
import hashlib
import uuid
from typing import Optional, Any
from src.backend.app.models.schemas import Session, UserProvidedImages, RetrievedImages
from langchain_core.messages import AIMessage


class SessionManager:

    def __init__(self):
        self._sessions: dict[str, Session] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> str:
        if session_id and session_id in self._sessions:
            return session_id
        elif session_id:
            self._sessions[session_id] = Session(image_source_store={}, image_ids_store=[], message_history=[])
            return session_id
        else:
            session_id = str(uuid.uuid4())
            self._sessions[session_id] = Session(image_source_store={}, image_ids_store=[], message_history=[])
            return session_id

    def get_model_source(self, session_id) -> ImageSource:
        model_image_id = self._sessions[session_id].model_image_id
        return self._sessions[session_id].image_source_store.get(model_image_id)

    def store_image_source(self, session_id: str, image_data: ImageSource, is_model: bool = False) -> str:
        key_string = f"{image_data.path}:{image_data.bbox}"
        unique_id = hashlib.md5(key_string.encode()).hexdigest()[:7]
        self._sessions[session_id].image_source_store[unique_id] = image_data
        if is_model:
            self._sessions[session_id].model_image_id = unique_id
        return unique_id

    def get_image_source(self, session_id: str, image_id: str) -> ImageSource:
        return self._sessions[session_id].image_source_store[image_id]

    def store_image_ids(self, session_id: str, image_ids: UserProvidedImages | RetrievedImages) -> None:
        self._sessions[session_id].image_ids_store.append(image_ids)

    def get_image_ids(self, session_id: str) -> list[UserProvidedImages | RetrievedImages]:
        return self._sessions[session_id].image_ids_store

    def load_message_history(self, session_id: str) -> list[AIMessage | dict[str, Any]]:
        return self._sessions[session_id].message_history

    def store_message(self, session_id: str, user_query: dict[str, Any], ai_response: AIMessage) -> None:
        self._sessions[session_id].message_history.append(user_query)
        self._sessions[session_id].message_history.append(ai_response)

    def cleanup_session(self, session_id: str):
        del self._sessions[session_id]
