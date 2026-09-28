"""Upload only corpus PDF passages to an existing S3 Vectors index.

This module never imports the checklist or changes query retrieval. Run with
``python -m langchain_backend.s3_ingestion --dry-run`` to inspect the corpus
without creating AWS clients, embedding text, or uploading anything.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math

import boto3
from botocore.config import Config
from .models import create_embeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .documents import CHUNK_OVERLAP, CHUNK_SIZE, load_pdf_documents
from .settings import settings

# Below both the embedding API's batch limit and S3 Vectors' 500-vector limit.
BATCH_SIZE = 64


def _index_destination() -> dict[str, str]:
    if settings.s3_vectors_index_arn:
        return {"indexArn": settings.s3_vectors_index_arn}
    if settings.s3_vectors_bucket_name and settings.s3_vectors_index_name:
        return {
            "vectorBucketName": settings.s3_vectors_bucket_name,
            "indexName": settings.s3_vectors_index_name,
        }
    raise ValueError(
        "Set S3_VECTORS_INDEX_ARN, or both S3_VECTORS_BUCKET_NAME and "
        "S3_VECTORS_INDEX_NAME, to an existing PDF corpus index. "
        "Use --dry-run to inspect the corpus without AWS configuration."
    )


def ingest_pdf_corpus(*, dry_run: bool = False) -> dict[str, int | str | bool]:
    """Embed and upsert corpus PDFs only; unchanged reruns reuse vector keys.

    Old corpus versions are not deleted. Use a dedicated index for each revised
    corpus, or filter by the returned corpus_version when adding retrieval.
    Errors propagate to the caller, including failures after a partial upload.
    """
    documents = load_pdf_documents(settings.corpus_dir)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = [chunk for chunk in splitter.split_documents(documents) if chunk.page_content.strip()]
    if not chunks:
        raise ValueError(f"No extractable PDF text found in {settings.corpus_dir}")

    records = []
    for position, chunk in enumerate(chunks):
        # Whitelist metadata: no local absolute paths or arbitrary PDF fields.
        metadata = {
            "source": chunk.metadata["source"],
            "page": int(chunk.metadata["page"]),  # Keep PyPDFLoader's zero-based pages.
            "chunk_index": position,
            "embedding_model": settings.embedding_model,
            "text": chunk.page_content,
        }
        key = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode("utf-8")).hexdigest()
        records.append({"key": key, "metadata": metadata})

    version = hashlib.sha256("".join(record["key"] for record in records).encode()).hexdigest()
    # Include the version in the key so revising one PDF cannot overwrite
    # unchanged passages that still belong to a previous corpus version.
    for record in records:
        record["key"] = hashlib.sha256(f"{version}:{record['key']}".encode()).hexdigest()
        record["metadata"]["corpus_version"] = version

    result: dict[str, int | str | bool] = {
        "target": "s3vectors",
        "dry_run": dry_run,
        "documents": len(documents),  # Existing API counts PDF pages here.
        "pdf_files": len({document.metadata["source"] for document in documents}),
        "chunks": len(chunks),
        "corpus_dir": str(settings.corpus_dir.resolve()),
        "corpus_version": version,
        "vectors_uploaded": 0,
    }
    if dry_run:
        return result

    destination = _index_destination()
    client = boto3.client(
        "s3vectors",
        region_name=settings.aws_region,
        config=Config(
            connect_timeout=10, read_timeout=60,
            retries={"mode": "standard", "total_max_attempts": 5},
        ),
    )
    index = client.get_index(**destination)["index"]
    non_filterable = set(index.get("metadataConfiguration", {}).get("nonFilterableMetadataKeys", []))
    if index["dataType"] != "float32" or "text" not in non_filterable:
        raise ValueError(
            "The S3 Vectors index must use float32 and declare 'text' as a "
            "non-filterable metadata key. Create a compatible index before ingesting."
        )
    if "corpus_version" in non_filterable:
        raise ValueError("The index must keep 'corpus_version' filterable.")

    # Validate the complete corpus before paying for any embedding calls.
    for record in records:
        metadata = record["metadata"]
        filterable = {key: value for key, value in metadata.items() if key not in non_filterable}
        if len(json.dumps(metadata, ensure_ascii=False).encode("utf-8")) > 40 * 1024:
            raise ValueError(f"PDF metadata exceeds 40 KB: {metadata['source']}")
        if len(json.dumps(filterable, ensure_ascii=False).encode("utf-8")) > 2 * 1024:
            raise ValueError(f"Filterable PDF metadata exceeds 2 KB: {metadata['source']}")

    embeddings = create_embeddings()
    result["index_arn"] = index["indexArn"]
    for offset in range(0, len(records), BATCH_SIZE):
        batch = records[offset:offset + BATCH_SIZE]
        # embed_documents uses the retrieval-document task type.
        vectors = embeddings.embed_documents([record["metadata"]["text"] for record in batch])
        if len(vectors) != len(batch):
            raise ValueError("Gemini returned a different number of vectors than PDF chunks.")
        payload = []
        for record, vector in zip(batch, vectors, strict=True):
            if len(vector) != index["dimension"]:
                raise ValueError(
                    f"Embedding dimension {len(vector)} does not match index dimension "
                    f"{index['dimension']}. Use the same embedding model and index dimensions."
                )
            values = [float(value) for value in vector]
            if not all(math.isfinite(value) and abs(value) <= 3.4028234663852886e38 for value in values):
                raise ValueError("Gemini returned an invalid float32 embedding.")
            if index["distanceMetric"] == "cosine" and not any(values):
                raise ValueError("A cosine index cannot accept an all-zero embedding.")
            payload.append({**record, "data": {"float32": values}})
        client.put_vectors(**destination, vectors=payload)
        result["vectors_uploaded"] = offset + len(batch)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest corpus PDFs only into S3 Vectors.")
    parser.add_argument("--dry-run", action="store_true", help="Count PDF pages/chunks without AWS calls.")
    args = parser.parse_args()
    print(json.dumps(ingest_pdf_corpus(dry_run=args.dry_run), indent=2))


if __name__ == "__main__":
    main()
