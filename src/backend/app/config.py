from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_model: str = "gpt-4.1"
    qdrant_url: str = "http://localhost:6333"
    clip_model_name: str = "patrickjohncyh/fashion-clip"
    openai_api_key: str

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
