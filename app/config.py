from pydantic import BaseModel


class Settings(BaseModel):
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Models
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim
    cache_threshold: float = 0.95
    retrieve_top_k: int = 4

    # Primary model (optional)
    primary_provider: str = "none"  # "openai" | "none"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4o-mini"

    # Local fallback (Ollama)
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3"


settings = Settings()
