from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_model: str = "gpt-4.1"
    vton_model: str = "gemini-3-pro-image-preview"
    qdrant_url: str = "http://localhost:6333"
    collection_name: str = "ctl_dataset_train_sample_500"
    clip_model_name: str = "patrickjohncyh/fashion-clip"
    openai_api_key: str
    google_api_key: str

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
