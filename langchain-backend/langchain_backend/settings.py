from __future__ import annotations

import os
from pathlib import Path


class Settings:
    def __init__(self) -> None:
        self.aws_region = os.getenv("AWS_REGION", "ap-southeast-1")
        self.bedrock_chat_model = os.getenv(
            "BEDROCK_CHAT_MODEL", "apac.amazon.nova-lite-v1:0"
        )
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
