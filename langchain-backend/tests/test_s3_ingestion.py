"""Offline ingestion tests: real PDF loading, fake Bedrock, validated S3 calls."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import boto3
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber
from langchain_core.documents import Document
from pypdf import PdfWriter
from pypdf.generic import DictionaryObject, NameObject, DecodedStreamObject

from langchain_backend import s3_ingestion
from langchain_backend.documents import load_pdf_documents
from langchain_backend.retrieval import RegulationSearch

INDEX_ARN = "arn:aws:s3vectors:ap-southeast-1:123456789012:bucket/test-corpus/index/mas-v1"


def write_pdf(path: Path, text: str) -> None:
    writer = PdfWriter()
    page = writer.add_blank_page(width=600, height=800)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font})
    })
    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 50 700 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = stream
    with path.open("wb") as file:
        writer.write(file)


class S3IngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        temp = TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "corpus"
        self.root.mkdir()
        (self.root / "nested").mkdir()
        write_pdf(self.root / "regulation.pdf", "PDF_REGULATION_ONLY")
        write_pdf(self.root / "nested" / "notice.PDF", "PDF_NOTICE_ONLY")
        write_pdf(self.root.parent / "outside.pdf", "OUTSIDE_CORPUS")
        (self.root / "checklist.py").write_text("CHECKLIST_MUST_NOT_BE_EMBEDDED")
        (self.root / "checklist.json").write_text('{"question":"DO_NOT_EMBED"}')
        self.addCleanup(patch.stopall)
        patch.multiple(
            s3_ingestion.settings,
            corpus_dir=self.root,
            aws_region="ap-southeast-1",
            bedrock_embedding_model="cohere.embed-english-v3",
            s3_vectors_index_arn=INDEX_ARN,
            s3_vectors_bucket_name="",
            s3_vectors_index_name="",
        ).start()
        self.client = boto3.client(
            "s3vectors", region_name="ap-southeast-1",
            aws_access_key_id="testing", aws_secret_access_key="testing",
        )
        self.stubber = Stubber(self.client)
        self.uploads = []
        self.client.meta.events.register(
            "before-parameter-build.s3vectors.PutVectors",
            lambda params, **kwargs: self.uploads.append(copy.deepcopy(params)),
        )
        self.client_factory = patch.object(s3_ingestion.boto3, "client", return_value=self.client).start()
        self.embeddings = Mock()
        self.embeddings.embed_documents.side_effect = lambda texts: [[1.0, 0.5, -0.25] for _ in texts]
        self.embedding_factory = patch.object(
            s3_ingestion, "BedrockEmbeddings", return_value=self.embeddings
        ).start()

    def queue_index(self, *, dimension=3, non_filterable=None, destination=None) -> None:
        self.stubber.add_response("get_index", {"index": {
            "indexArn": INDEX_ARN,
            "indexName": "mas-v1",
            "vectorBucketName": "test-corpus",
            "creationTime": datetime(2026, 9, 1, tzinfo=timezone.utc),
            "dataType": "float32",
            "dimension": dimension,
            "distanceMetric": "cosine",
            "metadataConfiguration": {
                "nonFilterableMetadataKeys": ["text"] if non_filterable is None else non_filterable,
            },
        }}, destination or {"indexArn": INDEX_ARN})

    def queue_upload(self, destination=None) -> None:
        self.stubber.add_response(
            "put_vectors", {}, {**(destination or {"indexArn": INDEX_ARN}), "vectors": ANY}
        )

    def test_dry_run_reads_only_corpus_pdfs_without_aws_or_credentials(self) -> None:
        with patch.object(s3_ingestion.settings, "s3_vectors_index_arn", ""):
            result = s3_ingestion.ingest_pdf_corpus(dry_run=True)
        self.assertEqual((result["pdf_files"], result["documents"], result["chunks"]), (2, 2, 2))
        self.assertEqual(result["vectors_uploaded"], 0)
        self.client_factory.assert_not_called()
        self.embedding_factory.assert_not_called()
        self.assertEqual(
            {doc.metadata["source"] for doc in load_pdf_documents(self.root)},
            {"regulation.pdf", "nested/notice.PDF"},
        )

    def test_pdf_upload_preserves_citations_and_identical_reruns_reuse_keys(self) -> None:
        for _ in range(2):
            self.queue_index()
            self.queue_upload()
        with self.stubber:
            first = s3_ingestion.ingest_pdf_corpus()
            second = s3_ingestion.ingest_pdf_corpus()
            self.stubber.assert_no_pending_responses()
        self.assertEqual(first["vectors_uploaded"], 2)
        self.assertEqual(first["corpus_version"], second["corpus_version"])
        self.assertEqual(self.uploads[0], self.uploads[1])
        metadata = [vector["metadata"] for vector in self.uploads[0]["vectors"]]
        self.assertEqual({m["text"].strip() for m in metadata}, {"PDF_REGULATION_ONLY", "PDF_NOTICE_ONLY"})
        for entry in metadata:
            self.assertEqual(entry["page"], 0)
            self.assertNotIn("path", entry)
            self.assertEqual(entry["corpus_version"], first["corpus_version"])
        self.embeddings.embed_query.assert_not_called()

    def test_revised_corpus_uses_new_keys_without_overwriting_old_version(self) -> None:
        for _ in range(2):
            self.queue_index()
            self.queue_upload()
        with self.stubber:
            first = s3_ingestion.ingest_pdf_corpus()
            write_pdf(self.root / "regulation.pdf", "UPDATED_REGULATION")
            second = s3_ingestion.ingest_pdf_corpus()
        self.assertNotEqual(first["corpus_version"], second["corpus_version"])
        old = {v["key"] for v in self.uploads[0]["vectors"]}
        new = {v["key"] for v in self.uploads[1]["vectors"]}
        self.assertTrue(old.isdisjoint(new))

    def test_uploads_are_batched_and_bucket_index_names_are_supported(self) -> None:
        destination = {"vectorBucketName": "test-corpus", "indexName": "mas-v1"}
        self.queue_index(destination=destination)
        for _ in range(3):
            self.queue_upload(destination)
        pages = [Document(page_content=f"Regulation {i}", metadata={"source": "regulation.pdf", "page": i})
                 for i in range(130)]
        with self.stubber, patch.multiple(
            s3_ingestion.settings, s3_vectors_index_arn="",
            s3_vectors_bucket_name="test-corpus", s3_vectors_index_name="mas-v1",
        ), patch.object(s3_ingestion, "load_pdf_documents", return_value=pages):
            result = s3_ingestion.ingest_pdf_corpus()
            self.stubber.assert_no_pending_responses()
        self.assertEqual([len(call["vectors"]) for call in self.uploads], [64, 64, 2])
        self.assertEqual(result["vectors_uploaded"], 130)

    def test_missing_destination_does_not_embed_or_upload(self) -> None:
        with patch.object(s3_ingestion.settings, "s3_vectors_index_arn", ""):
            with self.assertRaisesRegex(ValueError, "S3_VECTORS_INDEX_ARN"):
                s3_ingestion.ingest_pdf_corpus()
        self.client_factory.assert_not_called()
        self.embedding_factory.assert_not_called()

    def test_empty_corpus_fails_before_aws(self) -> None:
        with patch.object(s3_ingestion, "load_pdf_documents", return_value=[]):
            with self.assertRaisesRegex(ValueError, "No extractable PDF text"):
                s3_ingestion.ingest_pdf_corpus()
        self.client_factory.assert_not_called()

    def test_filterable_passage_text_rejected_before_embedding(self) -> None:
        self.queue_index(non_filterable=["unused"])
        with self.stubber, self.assertRaisesRegex(ValueError, "non-filterable"):
            s3_ingestion.ingest_pdf_corpus()
        self.embedding_factory.assert_not_called()

    def test_dimension_mismatch_never_uploads(self) -> None:
        self.queue_index(dimension=1024)
        with self.stubber, self.assertRaisesRegex(ValueError, "dimension"):
            s3_ingestion.ingest_pdf_corpus()
        self.assertEqual(self.uploads, [])

    def test_invalid_embeddings_never_upload(self) -> None:
        for vector in ([float("nan"), 1.0, 2.0], [0.0, 0.0, 0.0]):
            with self.subTest(vector=vector):
                self.queue_index()
                self.embeddings.embed_documents.side_effect = lambda texts: [vector for _ in texts]
                with self.stubber, self.assertRaises(ValueError):
                    s3_ingestion.ingest_pdf_corpus()
        self.assertEqual(self.uploads, [])

    def test_permission_error_is_reported_instead_of_success(self) -> None:
        self.stubber.add_client_error("get_index", service_error_code="AccessDeniedException")
        with self.stubber, self.assertRaises(ClientError):
            s3_ingestion.ingest_pdf_corpus()
        self.embedding_factory.assert_not_called()

    def test_failed_upload_is_reported_instead_of_success(self) -> None:
        self.queue_index()
        self.stubber.add_client_error("put_vectors", service_error_code="AccessDeniedException")
        with self.stubber, self.assertRaises(ClientError):
            s3_ingestion.ingest_pdf_corpus()

    def test_explicit_s3_ingest_does_not_replace_local_retrieval_store(self) -> None:
        search = RegulationSearch.__new__(RegulationSearch)
        local_store = object()
        search._store = local_store
        result = search.ingest(target="s3vectors", dry_run=True)
        self.assertEqual(result["target"], "s3vectors")
        self.assertIs(search.vector_store(), local_store)

    def test_ingest_api_exposes_s3_result_and_preserves_faiss_default(self) -> None:
        from importlib import import_module
        from fastapi.testclient import TestClient

        with patch("langchain_backend.rag.CodeRAGService"):
            main = import_module("langchain_backend.main")
        result = s3_ingestion.ingest_pdf_corpus(dry_run=True)
        with patch.object(main, "service") as service, TestClient(main.app) as client:
            service.ingest.return_value = result
            response = client.post("/ingest?target=s3vectors&dry_run=true")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json(), result)
            service.ingest.assert_called_once_with(target="s3vectors", dry_run=True)
            service.ingest.reset_mock()
            response = client.post("/ingest?target=unsupported")
            self.assertEqual(response.status_code, 422)
            service.ingest.assert_not_called()
            service.ingest.return_value = {
                "target": "faiss", "dry_run": False, "documents": 2,
                "pdf_files": 2, "chunks": 2, "corpus_dir": str(self.root),
                "index_dir": ".index",
            }
            response = client.post("/ingest")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["index_dir"], ".index")
            service.ingest.assert_called_once_with(target="faiss", dry_run=False)

    def test_missing_local_index_still_builds_faiss_only(self) -> None:
        search = RegulationSearch.__new__(RegulationSearch)
        search._store = None
        local_store = object()

        def build_local(**kwargs):
            self.assertEqual(kwargs, {})  # The existing implicit path keeps default FAISS.
            search._store = local_store

        with patch("langchain_backend.retrieval.load_vector_store", side_effect=FileNotFoundError), \
             patch.object(search, "ingest", side_effect=build_local):
            search._embeddings = Mock()
            self.assertIs(search.vector_store(), local_store)
        self.client_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
