from __future__ import annotations

import os
import hashlib
from pathlib import Path

from dotenv import load_dotenv

# `settings` below is a module-level singleton, so it is constructed the moment
# anything imports this module — which happens well before an entry point gets a
# chance to call `load_dotenv` itself. Loading here is what makes the `.env`
# values actually reach `Settings` rather than silently losing to the defaults.
load_dotenv(Path(__file__).resolve().parents[2] / ".env")


class Settings:
    def __init__(self) -> None:
        self.aws_region = os.getenv("AWS_REGION", "ap-southeast-1")
        self.chat_model = os.getenv("CLEARCALL_MODEL_ID", "gemini-3.5-flash")
        self.temperature = float(os.getenv("CLEARCALL_TEMPERATURE", "1"))
        self.embedding_model = os.getenv("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
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
        # Different models inhabit different vector spaces, even at equal dimensions.
        model_hash = hashlib.sha256(self.embedding_model.encode()).hexdigest()[:12]
        self.index_dir = self.index_dir / f"google-{model_hash}"
        # Ingestion destination only; query retrieval continues to use FAISS.
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
