from __future__ import annotations

import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

# `settings` below is a module-level singleton, so it is constructed the moment
# anything imports this module — which happens well before an entry point gets a
# chance to call `load_dotenv` itself. Loading here is what makes the `.env`
# values actually reach `Settings` rather than silently losing to the defaults.
load_dotenv(find_dotenv())


class Settings:
    def __init__(self) -> None:
        self.aws_region = os.getenv("AWS_REGION", "ap-southeast-1")
        self.bedrock_chat_model = os.getenv(
            "BEDROCK_CHAT_MODEL", "apac.amazon.nova-lite-v1:0"
        )
        # Titan embeddings are not offered in ap-southeast-1; Cohere v3 is the
        # on-demand English embedding model available there.
        self.bedrock_embedding_model = os.getenv(
            "BEDROCK_EMBEDDING_MODEL", "cohere.embed-english-v3"
        )
        self.workspace_root = Path(
            os.getenv("WORKSPACE_ROOT", str(Path(__file__).resolve().parents[2]))
        )
        self.corpus_dir = Path(
            os.getenv(
                "LANGCHAIN_CORPUS_DIR",
                str(Path(__file__).resolve().parents[1] / "corpus"),
            )
        )
        self.index_dir = Path(
            os.getenv(
                "LANGCHAIN_INDEX_DIR",
                str(Path(__file__).resolve().parents[1] / ".index"),
            )
        )
        self.regulation_search_backend = os.getenv("REGULATION_SEARCH_BACKEND", "faiss").strip().lower()
        if self.regulation_search_backend not in {"faiss", "s3vectors"}:
            raise ValueError("REGULATION_SEARCH_BACKEND must be faiss or s3vectors.")
        self.s3_vectors_corpus_version = os.getenv("S3_VECTORS_CORPUS_VERSION", "").strip()
        self.s3_vectors_index_arn = os.getenv("S3_VECTORS_INDEX_ARN", "").strip()
        self.s3_vectors_bucket_name = os.getenv("S3_VECTORS_BUCKET_NAME", "").strip()
        self.s3_vectors_index_name = os.getenv("S3_VECTORS_INDEX_NAME", "").strip()
        raw_extensions = os.getenv(
            "LANGCHAIN_ALLOWED_EXTENSIONS",
            ".py,.md,.txt,.json,.yaml,.yml,.js,.ts,.tsx,.jsx",
        )
        self.allowed_extensions = tuple(
            ext.strip().lower() for ext in raw_extensions.split(",") if ext.strip()
        )
        self.max_file_size_bytes = int(
            os.getenv("LANGCHAIN_MAX_FILE_SIZE_BYTES", "1048576")
        )


settings = Settings()
