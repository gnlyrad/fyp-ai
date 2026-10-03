"""Offline retrieval contract tests; no account credentials or AWS calls."""
import os
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'

import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timezone
import boto3
from botocore.stub import Stubber
from langchain_backend import s3_retrieval, retrieval

ARN = 'arn:aws:s3vectors:ap-southeast-1:123456789012:bucket/test/index/mas'


class S3RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.config = patch.multiple(s3_retrieval.settings,
            s3_vectors_index_arn=ARN, s3_vectors_corpus_version='version1',
            embedding_model='gemini-embedding-001', regulation_search_backend='s3vectors')
        self.config.start()
        self.addCleanup(self.config.stop)
        self.client = boto3.client('s3vectors', region_name='ap-southeast-1',
                                   aws_access_key_id='testing', aws_secret_access_key='testing')
        self.stub = Stubber(self.client)
        self.stub.activate()
        self.addCleanup(self.stub.deactivate)
        factory = patch.object(s3_retrieval.boto3, 'client', return_value=self.client)
        factory.start()
        self.addCleanup(factory.stop)
        self.embeddings = Mock()
        self.embeddings.embed_query.return_value = [1.0, 0.5, -0.25]

    def reader(self):
        self.stub.add_response('get_index', {'index': {
            'indexArn': ARN, 'indexName': 'mas', 'vectorBucketName': 'test',
            'creationTime': datetime.now(timezone.utc), 'dataType': 'float32',
            'dimension': 3, 'distanceMetric': 'cosine',
            'metadataConfiguration': {'nonFilterableMetadataKeys': ['text']}}}, {'indexArn': ARN})
        return s3_retrieval.S3RegulationSearch(self.embeddings)

    def expected_query(self):
        return dict(indexArn=ARN, queryVector={'float32': [1.0, 0.5, -0.25]}, topK=3,
            filter={'$and': [{'corpus_version': {'$eq': 'version1'}},
                            {'embedding_model': {'$eq': 'gemini-embedding-001'}}]},
            returnMetadata=True, returnDistance=True)

    def hit(self):
        return {'key': 'chunk1', 'distance': 0.2, 'metadata': {
            'text': 'Disclose all fees.', 'source': 'notice.pdf', 'page': 0,
            'corpus_version': 'version1', 'embedding_model': 'gemini-embedding-001'}}

    def test_query_preserves_text_citations_and_filters_version(self):
        reader = self.reader()
        self.stub.add_response('query_vectors', {'vectors': [self.hit()], 'distanceMetric': 'cosine'}, self.expected_query())
        documents = reader.search('fees', 3)
        self.assertEqual(documents[0].page_content, 'Disclose all fees.')
        self.assertEqual(documents[0].metadata['page'], 0)
        self.assertEqual(documents[0].metadata['retrieval_backend'], 's3vectors')
        self.embeddings.embed_query.assert_called_once_with('fees')
        self.stub.assert_no_pending_responses()

    def test_missing_version_fails_before_aws(self):
        with patch.object(s3_retrieval.settings, 's3_vectors_corpus_version', ''):
            with self.assertRaisesRegex(ValueError, 'S3_VECTORS_CORPUS_VERSION'):
                s3_retrieval.S3RegulationSearch(self.embeddings)

    def test_dimension_and_invalid_vectors_rejected_before_query(self):
        reader = self.reader()
        for vector in [[1.0], [0, 0, 0], [float('nan'), 0, 1]]:
            with self.subTest(vector=vector):
                self.embeddings.embed_query.return_value = vector
                with self.assertRaises(ValueError):
                    reader.search('fees', 3)

    def test_empty_version_or_corrupt_metadata_not_silent(self):
        reader = self.reader()
        invalid = self.hit()
        invalid['metadata']['corpus_version'] = 'old'
        for hits in [[], [invalid]]:
            self.stub.add_response('query_vectors', {'vectors': hits, 'distanceMetric': 'cosine'}, self.expected_query())
            with self.assertRaises(ValueError):
                reader.search('fees', 3)

    def test_aws_denial_propagates(self):
        from botocore.exceptions import ClientError
        reader = self.reader()
        self.stub.add_client_error('query_vectors', service_error_code='AccessDeniedException',
                                   expected_params=self.expected_query())
        with self.assertRaises(ClientError):
            reader.search('fees', 3)

    def test_backend_selection_does_not_use_faiss_for_s3(self):
        with patch.object(retrieval, 'create_embeddings', return_value=self.embeddings):
            search = retrieval.RegulationSearch()
        with patch.object(s3_retrieval, 'S3RegulationSearch') as factory, \
             patch.object(search, 'vector_store') as faiss:
            factory.return_value.search.return_value = []
            search.search('fees', 3)
            faiss.assert_not_called()
            factory.return_value.search.assert_called_once_with('fees', 3)
            with patch.object(retrieval.settings, 'regulation_search_backend', 'faiss'):
                search.search('fees', 3)
                faiss.return_value.similarity_search.assert_called_once_with('fees', k=3)

    def test_search_endpoint_returns_backend_and_citations_without_chat(self):
        from importlib import import_module
        from fastapi.testclient import TestClient
        from langchain_core.documents import Document
        with patch('langchain_backend.rag.CodeRAGService'), \
             patch.object(retrieval, 'create_embeddings', return_value=self.embeddings):
            main = import_module('langchain_backend.main')
        with patch.object(main, 'search_service') as search, \
             patch.object(main, 'service') as chat, TestClient(main.app) as client:
            search.search.return_value = [Document(page_content='Disclose fees.',
                metadata={'source': 'notice.pdf', 'page': 0, 'corpus_version': 'version1'})]
            response = client.post('/search', json={'question': 'fees', 'top_k': 3})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()['backend'], 's3vectors')
            self.assertEqual(response.json()['passages'][0]['source'], 'notice.pdf')
            search.search.assert_called_once_with('fees', k=3)
            chat.answer.assert_not_called()
            self.assertEqual(client.post('/search', json={'question': 'fees', 'top_k': 0}).status_code, 422)


if __name__ == '__main__':
    unittest.main()
