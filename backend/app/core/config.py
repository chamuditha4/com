"""12-factor configuration.

Every tunable lives here and is read from environment variables (or `.env` for local
development). Nothing else in the codebase reads `os.environ` directly, so the effective
configuration of a running worker is always inspectable in one place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", case_sensitive=False)

    # --- Application -------------------------------------------------------------------
    app_env: Literal["local", "test", "production"] = "local"
    app_name: str = "Commercial Bank Enterprise AI Assistant"
    log_level: str = "INFO"
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:8501"])

    # --- Auth ----------------------------------------------------------------------------
    jwt_secret: SecretStr = SecretStr("change-me-in-env-this-is-only-for-local-dev")
    jwt_algorithm: Literal["HS256"] = "HS256"
    jwt_ttl_minutes: int = 60

    # --- Shared state (Redis) ----------------------------------------------------------
    # When unset, workers fall back to in-process implementations. That is fine for a
    # single-process demo but NOT for horizontal scaling (see docs/ASSUMPTIONS.md).
    redis_url: str | None = None

    # --- Rate limiting (token bucket, per user) ---------------------------------------
    rate_limit_enabled: bool = True
    rate_limit_capacity: int = 10
    rate_limit_refill_per_second: float = 0.2  # 1 token every 5s => 12 requests/min sustained

    # --- LLM ---------------------------------------------------------------------------
    llm_provider: Literal["anthropic", "openai", "gemini", "none"] = "anthropic"
    llm_model: str = "claude-sonnet-5"  # reasoning tier: planning, synthesis
    llm_fast_model: str = "claude-haiku-4-5-20251001"  # fast tier: routing, extraction
    llm_fallback_provider: Literal["anthropic", "openai", "gemini", "none"] = "none"
    llm_fallback_model: str = ""
    llm_temperature: float = 0.0
    llm_timeout_seconds: float = 60.0
    llm_max_retries: int = 2
    llm_max_concurrency: int = 8
    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    google_api_key: SecretStr | None = None

    # --- Embeddings --------------------------------------------------------------------
    embedding_provider: Literal["pinecone", "openai", "hash"] = "hash"
    embedding_model: str = "multilingual-e5-large"
    embedding_dimension: int = 1024

    # --- Vector store ------------------------------------------------------------------
    vector_store: Literal["pinecone", "memory"] = "memory"
    pinecone_api_key: SecretStr | None = None
    pinecone_dense_index: str = "cb-knowledge-dense"
    pinecone_sparse_index: str = "cb-knowledge-sparse"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- Retrieval -----------------------------------------------------------------------
    retrieval_candidates_per_leg: int = 20
    retrieval_fusion: Literal["rrf", "weighted"] = "rrf"
    retrieval_dense_weight: float = 0.6
    retrieval_sparse_weight: float = 0.4
    retrieval_rrf_k: int = 60
    retrieval_top_n: int = 6
    retrieval_namespace_concurrency: int = 6
    reranker: Literal["pinecone", "lexical", "none"] = "lexical"
    reranker_model: str = "bge-reranker-v2-m3"

    # --- Recursive research (RLM) --------------------------------------------------------
    rlm_max_batches: int = 8
    rlm_max_depth: int = 2
    # Deliberately small so recursion is visible on the demo corpus; raise for production corpora.
    rlm_max_docs_per_slice: int = 2
    rlm_chunks_per_slice: int = 12
    rlm_batch_concurrency: int = 4

    # --- Agent -----------------------------------------------------------------------------
    agent_max_validation_retries: int = 1
    agent_max_tool_iterations: int = 3
    agent_history_window: int = 10  # messages kept verbatim in short-term memory
    chat_timeout_seconds: float = 180.0  # upper bound for one turn, including RLM fan-out

    # --- Memory ----------------------------------------------------------------------------
    checkpointer: Literal["memory", "redis"] = "memory"
    session_ttl_minutes: int = 24 * 60
    long_term_memory_enabled: bool = True

    # --- Tools -----------------------------------------------------------------------------
    mcp_server_url: str = "http://localhost:8765/mcp"
    mcp_timeout_seconds: float = 10.0
    python_tool_timeout_seconds: float = 5.0
    python_tool_memory_mb: int = 256

    # --- Data artifacts --------------------------------------------------------------------
    mock_docs_dir: Path = REPO_ROOT / "data" / "mock"
    artifacts_dir: Path = REPO_ROOT / "data" / "artifacts"

    # --- Observability ----------------------------------------------------------------------
    # LangSmith itself is configured through its own env vars (LANGSMITH_TRACING,
    # LANGSMITH_API_KEY, LANGSMITH_PROJECT); we only keep what the UI needs to deep-link.
    langsmith_project: str = "enterprise-ai-assistant"
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None
    langsmith_endpoint: str = "https://api.smith.langchain.com"

    @model_validator(mode="after")
    def _check_production_safety(self) -> Settings:
        if self.app_env == "production":
            if "change-me" in self.jwt_secret.get_secret_value():
                raise ValueError("JWT_SECRET must be set to a strong secret in production")
            if not self.redis_url:
                raise ValueError("REDIS_URL is required in production (stateless workers)")
        return self

    @property
    def bm25_params_path(self) -> Path:
        return self.artifacts_dir / "bm25_params.json"

    @property
    def catalog_path(self) -> Path:
        return self.artifacts_dir / "catalog.json"


@lru_cache
def get_settings() -> Settings:
    return Settings()
