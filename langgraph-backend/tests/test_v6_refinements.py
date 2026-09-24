"""Offline regression cases for the V6 classification/report issues."""
import json
import unittest
from unittest.mock import Mock, patch

from test_reliability import task_for
from compliance_pipeline.nodes import analysis, classification
from compliance_pipeline.nodes.output import output_node


def entry(item_id, condition, query=''):
    return classification.ItemType(id=item_id, question_type='deterministic',
        assessment='presence', rationale='Required disclosure.',
        pass_condition=condition, evidence_query=query)


class V6RefinementTests(unittest.TestCase):
    def test_retries_only_invalid_entries_and_recovers(self):
        model = Mock()
        invoke = model.with_structured_output.return_value.invoke
        invoke.side_effect = [
            classification.ChecklistClassification(items=[
                entry('fees', 'pass'), entry('product', 'The RM identifies the product structure.', 'product structure')]),
            classification.ChecklistClassification(items=[entry('fees', 'The RM discloses the applicable fees.', 'fees charges disclosure')])]
        with patch.object(classification, 'get_model', return_value=model):
            items = classification.classification_node({'checklist': [
                {'id': 'fees', 'question': 'Did the RM disclose fees?'},
                {'id': 'product', 'question': 'Did the RM identify the product structure?'}]})['checklist']
        self.assertEqual(invoke.call_count, 2)
        retried = json.loads(invoke.call_args_list[1].args[0][1].content)
        self.assertEqual([i['id'] for i in retried], ['fees'])
        self.assertTrue(retried[0]['validation_feedback'])
        self.assertTrue(all(i['classification_status'] == 'validated' for i in items))
        self.assertEqual(items[0]['evidence_query_source'], 'classifier')

    def test_invalid_caller_condition_retries_and_unresolved_na_requires_review(self):
        task = task_for()
        task['item']['pass_condition'] = 'pass'
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = classification.ChecklistClassification(
            items=[entry('fees', 'All requirements are met.', 'N/A')])
        with patch.object(classification, 'get_model', return_value=model):
            task['item'] = classification.classification_node({'checklist': [task['item']]})['checklist'][0]
        self.assertEqual(task['item']['classification_attempts'], 2)
        self.assertEqual(task['item']['classification_status'], 'needs_review')
        self.assertEqual(task['item']['pass_condition'], '')
        self.assertEqual(task['item']['evidence_query'], 'disclose fees')
        task['applicability'] = {'status': 'not_applicable', 'reason': 'Unrelated conversation.',
                                 'applicability_coverage': 'full_transcript'}
        with patch.object(analysis, 'judge_model') as judge:
            result = analysis._run_analysis(task)
        judge.assert_not_called()
        finding = result['raw_findings'][0]
        self.assertEqual(finding['verdict'], 'N/A')
        self.assertTrue(finding['requires_review'])
        summary = output_node({**result, 'consistency': {'status': 'not_applicable'}})['summary']
        self.assertTrue(summary['requires_review'])

    def test_query_fallback_does_not_require_reclassifying_valid_metadata(self):
        for query in ['', 'pass', 'N/A', 'search query']:
            with self.subTest(query=query), patch.object(classification, 'get_model') as model:
                item = {**task_for()['item'], 'evidence_query': query}
                result = classification.classification_node({'checklist': [item]})['checklist'][0]
                model.assert_not_called()
                self.assertEqual(result['evidence_query'], 'disclose fees')
                self.assertEqual(result['evidence_query_source'], 'question_fallback')

    def test_model_failure_is_bounded_and_applicable_finding_is_unclear(self):
        model = Mock()
        model.with_structured_output.return_value.invoke.side_effect = RuntimeError('offline')
        task = task_for()
        task['item'].pop('pass_condition')
        with patch.object(classification, 'get_model', return_value=model):
            task['item'] = classification.classification_node({'checklist': [task['item']]})['checklist'][0]
        self.assertEqual(model.with_structured_output.return_value.invoke.call_count, 2)
        finding = analysis._run_analysis(task)['raw_findings'][0]
        self.assertEqual(finding['evidence_status'], 'classification_unavailable')
        self.assertEqual(finding['verdict'], 'Unclear')

    def test_unknown_roles_warn_without_blocking_na_but_block_applicable(self):
        task = task_for()
        for chunk in task['chunks']:
            chunk.pop('speaker_role', None)
        warning = 'Speaker identities are preserved; some RM/Customer roles are unverified.'
        task['applicability'] = {'status': 'not_applicable', 'reason': 'Unrelated conversation.',
                                 'applicability_coverage': 'full_transcript'}
        result = analysis._run_analysis(task)
        finding = result['raw_findings'][0]
        self.assertEqual(finding['applicability_coverage'], 'full_transcript')
        self.assertEqual(finding['analysis_coverage'], 'skipped_not_applicable')
        self.assertNotIn('coverage', finding)
        summary = output_node({**result, 'input_warnings': [warning],
                              'consistency': {'status': 'not_applicable'}})['summary']
        self.assertEqual(summary['warnings'], [warning])
        self.assertFalse(summary['requires_review'])
        task['applicability']['status'] = 'applicable'
        result = analysis._run_analysis(task)
        self.assertEqual(result['raw_findings'][0]['evidence_status'], 'speaker_role_unverified')
        summary = output_node({**result, 'input_warnings': [warning],
                              'consistency': {'status': 'checked'}})['summary']
        self.assertTrue(summary['requires_review'])


if __name__ == '__main__':
    unittest.main()
