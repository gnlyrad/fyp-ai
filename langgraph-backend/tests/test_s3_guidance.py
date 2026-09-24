"""The graph's regulatory tool must expose explicit S3 backend failures."""
import os
os.environ['LANGSMITH_TRACING'] = 'false'
os.environ['LANGCHAIN_TRACING_V2'] = 'false'
import unittest
from unittest.mock import patch
from langchain_core.documents import Document
from compliance_pipeline import knowledge


class S3GuidanceTests(unittest.TestCase):
    def test_s3_failure_propagates_but_faiss_retains_optional_guidance(self):
        with patch.object(knowledge, 'CACHE_ENABLED', False), \
             patch.object(knowledge, '_regulation_search', side_effect=RuntimeError('AWS unavailable')):
            with patch.dict(os.environ, {'REGULATION_SEARCH_BACKEND': 's3vectors'}):
                with self.assertRaisesRegex(RuntimeError, 'AWS unavailable'):
                    knowledge.search_regulations.invoke({'query': 'fees'})
            with patch.dict(os.environ, {'REGULATION_SEARCH_BACKEND': 'faiss'}):
                self.assertEqual(knowledge.search_regulations.invoke({'query': 'fees'}), [])

    def test_document_citations_reach_guidance_with_one_based_page(self):
        with patch.object(knowledge, 'CACHE_ENABLED', False), \
             patch.object(knowledge, '_regulation_search') as search:
            search.return_value.search.return_value = [Document(page_content='Disclose fees.',
                metadata={'source': 'notice.pdf', 'page': 0, 'retrieval_backend': 's3vectors'})]
            result = knowledge.search_regulations.invoke({'query': 'fees'})
        self.assertEqual(result, [{'text': 'Disclose fees.', 'source': 'notice.pdf', 'page': 1}])
