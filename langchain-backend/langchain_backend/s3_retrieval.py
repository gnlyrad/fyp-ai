"""Query a successfully ingested, explicitly selected regulatory corpus version."""
from __future__ import annotations

import math

import boto3
from botocore.config import Config
from langchain_core.documents import Document

from .s3_ingestion import _index_destination
from .settings import settings


class S3RegulationSearch:
    def __init__(self, embeddings):
        self.destination = _index_destination()
        self.version = settings.s3_vectors_corpus_version
        if not self.version:
            raise ValueError("Set S3_VECTORS_CORPUS_VERSION to the corpus_version from a successful ingestion.")
        self.model = settings.bedrock_embedding_model
        self.embeddings = embeddings
        self.client = boto3.client("s3vectors", region_name=settings.aws_region,
            config=Config(connect_timeout=10, read_timeout=60,
                          retries={"mode": "standard", "total_max_attempts": 3}))
        self.index = self.client.get_index(**self.destination)["index"]
        non_filterable = set(self.index.get("metadataConfiguration", {}).get("nonFilterableMetadataKeys", []))
        if self.index["dataType"] != "float32" or self.index["distanceMetric"] != "cosine":
            raise ValueError("Regulatory retrieval requires a float32, cosine S3 Vectors index.")
        if "text" not in non_filterable or non_filterable.intersection({"corpus_version", "embedding_model"}):
            raise ValueError("Index must declare text non-filterable and keep corpus_version/embedding_model filterable.")

    def search(self, query: str, k: int) -> list[Document]:
        vector = [float(value) for value in self.embeddings.embed_query(query)]
        if len(vector) != self.index["dimension"]:
            raise ValueError("Query embedding dimension does not match the S3 Vectors index.")
        if not any(vector) or not all(math.isfinite(v) and abs(v) <= 3.4028234663852886e38 for v in vector):
            raise ValueError("Query embedding must contain finite float32 values and be nonzero.")
        response = self.client.query_vectors(
            **self.destination, queryVector={"float32": vector}, topK=k,
            filter={"$and": [{"corpus_version": {"$eq": self.version}},
                             {"embedding_model": {"$eq": self.model}}]},
            returnMetadata=True, returnDistance=True)
        hits = response.get("vectors", [])
        if not hits:
            raise ValueError("No vectors matched the configured corpus version and embedding model; verify ingestion completed.")
        documents = []
        for hit in hits:
            metadata = hit.get("metadata", {})
            text, source, page = metadata.get("text"), metadata.get("source"), metadata.get("page")
            if (not isinstance(text, str) or not text.strip() or not isinstance(source, str) or not source
                    or isinstance(page, bool) or not isinstance(page, (int, float))
                    or not math.isfinite(page) or page < 0 or int(page) != page
                    or metadata.get("corpus_version") != self.version
                    or metadata.get("embedding_model") != self.model):
                raise ValueError("S3 Vectors returned invalid regulatory text, citation metadata, or corpus version.")
            documents.append(Document(page_content=text, metadata={
                "source": source, "page": int(page), "vector_key": hit["key"],
                "distance": hit["distance"], "retrieval_backend": "s3vectors",
                "corpus_version": self.version, "embedding_model": self.model}))
        return documents
