"""Regression tests for recovery, context expansion, and failure reporting."""
import os
import unittest
from unittest.mock import Mock, patch

from botocore.exceptions import ClientError

from compliance_pipeline import reliability
from compliance_pipeline.nodes import analysis, applicability, classification, consistency, guidance, retrieval
from compliance_pipeline.nodes.output import output_node
from test_reliability import auditor, judgement, task_for


class PredictionRecoveryTests(unittest.TestCase):
    def test_classification_auth_failure_is_not_retried_and_reaches_finding(self):
        task = task_for()
        task["item"].pop("pass_condition")
        model = Mock()
        model.with_structured_output.return_value.invoke.side_effect = ClientError(
            {"Error": {"Code": "UnrecognizedClientException", "Message": "sensitive provider detail"}},
            "Converse",
        )
        with patch.object(classification, "get_model", return_value=model):
            task["item"] = classification.classification_node({"checklist": [task["item"]]})["checklist"][0]
        self.assertEqual(model.with_structured_output.return_value.invoke.call_count, 1)
        with patch.object(analysis, "judge_model") as judge:
            finding = analysis._run_analysis(task)["findings"][0]
        judge.assert_not_called()
        self.assertIsNone(finding["verdict"])
        self.assertEqual(finding["failure_error_code"], "UnrecognizedClientException")
        self.assertEqual(finding["failure_error_type"], "ClientError")
        self.assertEqual(finding["failure_stage"], "classification")
        self.assertNotIn("sensitive provider detail", str(finding))

    def test_classification_transient_failure_recovers_without_stale_failure(self):
        item = {"id": "fees", "question": "Did the RM disclose fees?"}
        model = Mock()
        model.with_structured_output.return_value.invoke.side_effect = [
            ClientError({"Error": {"Code": "ThrottlingException"}}, "Converse"),
            classification.ChecklistClassification(items=[classification.ItemType(
                id="fees", question_type="deterministic", rationale="Explicit disclosure",
                pass_condition="The RM discloses the applicable fees.", assessment="presence",
                evidence_query="fees disclosure",
            )]),
        ]
        with patch.object(classification, "get_model", return_value=model):
            result = classification.classification_node({"checklist": [item]})["checklist"][0]
        self.assertEqual(model.with_structured_output.return_value.invoke.call_count, 2)
        self.assertEqual(result["classification_status"], "validated")
        self.assertNotIn("classification_failure", result)
        self.assertNotIn("classification_error", result)

    def setUp(self):
        self.sleep = patch.object(reliability.time, "sleep").start()
        self.addCleanup(patch.stopall)

    def run_judge(self, responses, task=None):
        model = Mock()
        model.invoke.side_effect = responses
        with patch.object(analysis, "judge_model", return_value=model), \
             patch.object(analysis, "get_model", return_value=auditor()):
            finding = analysis._run_analysis(task or task_for())["findings"][0]
        return finding, model

    def test_transient_judge_error_recovers(self):
        finding, model = self.run_judge([TimeoutError(), judgement()])
        self.assertEqual(finding["verdict"], "Pass")
        self.assertEqual(model.invoke.call_count, 2)
        self.assertTrue(finding["analysis_diagnostics"][0]["recovered"])
        self.sleep.assert_called_once()

    def test_throttling_retries_but_invalid_credentials_do_not(self):
        for code, attempts in [("ThrottlingException", 3), ("UnrecognizedClientException", 1),
                               ("AccessDeniedException", 1), ("ValidationException", 1)]:
            with self.subTest(code=code):
                error = ClientError({"Error": {"Code": code, "Message": "test"}}, "Converse")
                finding, model = self.run_judge([error] * 3)
                self.assertEqual(model.invoke.call_count, attempts)
                self.assertIsNone(finding["verdict"])
                self.assertEqual(finding["execution_status"], "prediction_failure")
                self.assertEqual(finding["failure_stage"], "judge")
                self.assertEqual(finding["failure_error_code"], code)
                self.assertEqual(finding["spans"], [])

    def test_malformed_output_gets_one_repair_with_feedback(self):
        finding, model = self.run_judge([None, judgement()])
        self.assertEqual(finding["verdict"], "Pass")
        self.assertIn("Output validation failed", model.invoke.call_args.args[0][-1].content)
        finding, model = self.run_judge([None, None, judgement()])
        self.assertEqual(model.invoke.call_count, 2)
        self.assertIsNone(finding["verdict"])

    def test_one_attempt_disables_recovery(self):
        task = {**task_for(), "analysis_max_attempts": 1}
        finding, model = self.run_judge([TimeoutError(), judgement()], task)
        self.assertEqual(model.invoke.call_count, 1)
        self.assertIsNone(finding["verdict"])
        self.sleep.assert_not_called()

    def test_audit_retry_preserves_judgement(self):
        model = auditor()
        audit = model.with_structured_output.return_value.invoke
        good = audit.return_value
        audit.side_effect = [TimeoutError(), good]
        with patch.object(analysis, "_invoke", return_value=judgement()) as judge, \
             patch.object(analysis, "get_model", return_value=model):
            finding = analysis._run_analysis(task_for())["findings"][0]
        judge.assert_called_once()
        self.assertEqual(audit.call_count, 2)
        self.assertEqual(finding["verdict"], "Pass")
        self.assertTrue(finding["analysis_diagnostics"][0]["recovered"])

    def test_exhausted_audit_does_not_publish_unaudited_spans(self):
        model = auditor()
        model.with_structured_output.return_value.invoke.side_effect = TimeoutError()
        with patch.object(analysis, "_invoke", return_value=judgement()) as judge, \
             patch.object(analysis, "get_model", return_value=model):
            finding = analysis._run_analysis(task_for("targeted"))["findings"][0]
        judge.assert_called_once()  # operational failure is not semantic fallback
        self.assertEqual(finding["failure_stage"], "evidence_audit")
        self.assertIsNone(finding["verdict"])
        self.assertEqual(finding["spans"], [])

    def test_incomplete_or_duplicate_audit_is_repaired(self):
        for invalid in [analysis.EvidenceAudit(), analysis.EvidenceAudit(items=[
                {"index": 0, "relation": "supports", "label": "compliant", "reason": "valid"}] * 2)]:
            model = auditor()
            invoke = model.with_structured_output.return_value.invoke
            invoke.side_effect = [invalid, invoke.return_value]
            with patch.object(analysis, "_invoke", return_value=judgement()), \
                 patch.object(analysis, "get_model", return_value=model):
                finding = analysis._run_analysis(task_for())["findings"][0]
            self.assertEqual(finding["verdict"], "Pass")
            self.assertEqual(invoke.call_count, 2)
            self.assertIn("every quote index", invoke.call_args.args[0][-1].content)

    def test_completed_full_call_unclear_is_not_retried(self):
        finding, model = self.run_judge([judgement("Unclear", "insufficient", [])])
        self.assertEqual(model.invoke.call_count, 1)
        self.assertEqual(finding["verdict"], "Unclear")
        self.assertEqual(finding["execution_status"], "success")

    def test_audit_rejected_targeted_pass_gets_full_call_second_look(self):
        task = task_for("targeted")
        model = auditor()
        invoke = model.with_structured_output.return_value.invoke
        supported = invoke.return_value
        invoke.side_effect = [analysis.EvidenceAudit(items=[{
            "index": 0, "relation": "irrelevant", "label": "context", "reason": "Insufficient context."}]), supported]
        with patch.object(analysis, "_invoke", return_value=judgement()) as judge, \
             patch.object(analysis, "get_model", return_value=model):
            finding = analysis._run_analysis(task)["findings"][0]
        self.assertEqual(judge.call_count, 2)
        self.assertEqual(judge.call_args.args[2], "full_transcript")
        self.assertEqual(finding["verdict"], "Pass")
        self.assertEqual(finding["analysis_coverage"], "full_transcript")

    def test_invalid_quote_gets_one_second_look_and_remains_unclear(self):
        answer = judgement(evidence=[{"chunk_id": "chunk-00", "quote": "fabricated", "label": "compliant"}])
        with patch.object(analysis, "_invoke", return_value=answer) as judge, \
             patch.object(analysis, "audit_evidence") as audit:
            finding = analysis._run_analysis(task_for("targeted"))["findings"][0]
        self.assertEqual(judge.call_count, 2)
        audit.assert_not_called()
        self.assertEqual(finding["verdict"], "Unclear")

    def test_full_context_preserves_type_and_skips_transcript_embeddings(self):
        task = {**task_for("targeted"), "context_mode": "full_transcript"}
        state = {"chunks": task["chunks"], "checklist": [task["item"]], "context_mode": "full_transcript"}
        with patch.object(retrieval, "embedding_client") as embed:
            result = retrieval.retrieval_node(state)
        embed.assert_not_called()
        self.assertEqual(result["evidence"]["fees"], [{**c, "score": 1.0} for c in task["chunks"]])
        with patch.object(analysis, "_invoke", return_value=judgement()) as judge, \
             patch.object(analysis, "get_model", return_value=auditor()):
            finding = analysis._run_analysis(task)["findings"][0]
        self.assertEqual(judge.call_args.args[2], "full_transcript")
        self.assertEqual(finding["question_type"], "targeted")

    def test_public_graph_input_propagates_full_context_and_retry_budget(self):
        from compliance_pipeline.graph import graph
        task = task_for("targeted")
        segments = [{"transcript": c["text"], "speaker_label": c["speaker_label"],
                     "start_time": str(c["start_seconds"]), "end_time": str(c["end_seconds"])}
                    for c in task["chunks"]]
        judge = Mock()
        judge.invoke.side_effect = TimeoutError()
        checker = Mock()
        checker.with_structured_output.return_value.invoke.return_value = consistency.ConsistencyReview()
        with patch.dict(os.environ, {"CLEARCALL_SKIP_APPLICABILITY": "true"}), \
             patch.object(analysis, "judge_model", return_value=judge), \
             patch.object(retrieval, "embedding_client") as embed, \
             patch.object(guidance, "search_regulations") as regulations, \
             patch.object(consistency, "get_model", return_value=checker):
            regulations.invoke.return_value = []
            result = graph.invoke({"audio_segments": segments, "checklist": [task["item"]],
                                   "speaker_roles": {"spk_0": "RM", "spk_1": "Customer"},
                                   "context_mode": "full_transcript", "analysis_max_attempts": 1})
        judge.invoke.assert_called_once()
        embed.assert_not_called()
        regulations.invoke.assert_called_once()  # guidance remains eligible
        finding = result["report"]["findings"][0]
        self.assertEqual(finding["analysis_coverage"], "full_transcript")
        self.assertEqual(finding["question_type"], "targeted")
        self.assertEqual(result["report"]["summary"]["prediction_failure"], 1)
        self.assertEqual(result["report"]["summary"]["unclear"], 0)

    def test_employee_requirement_rejects_customer_evidence(self):
        task = task_for()
        task["item"].pop("subject_role")
        task["item"]["question"] = "The employee must disclose the bank's transaction fee."
        task["item"] = classification.classification_node({"checklist": [task["item"]]})["checklist"][0]
        self.assertEqual(task["item"]["subject_role"], "RM")
        answer = judgement(evidence=[{"chunk_id": "chunk-01", "quote": "I understand.", "label": "compliant"}])
        with patch.object(analysis, "_invoke", return_value=answer), \
             patch.object(analysis, "get_model", return_value=auditor()):
            finding = analysis._run_analysis(task)["findings"][0]
        self.assertEqual(finding["verdict"], "Unclear")
        self.assertEqual(finding["spans"][0]["label"], "context")

    def test_valid_caller_role_is_preserved_and_invalid_role_rejected(self):
        item = {**task_for()["item"], "subject_role": "Customer"}
        self.assertEqual(classification.classification_node({"checklist": [item]})["checklist"][0]["subject_role"], "Customer")
        with self.assertRaises(ValueError):
            classification.classification_node({"checklist": [{**item, "subject_role": "employee"}]})

    def test_employee_as_recipient_does_not_override_client_actor(self):
        self.assertEqual(classification.subject_role(
            "Which investment instrument does the client instruct the employee to transact in?"), "Customer")
        self.assertEqual(classification.subject_role(
            "Did the client acknowledge the RM's explanation?"), "Customer")
        self.assertEqual(classification.subject_role(
            "The employee must confirm the client's willingness to accept investment losses."), "RM")

    def test_api_counts_failures_separately_including_legacy_outputs(self):
        findings = [
            {"checklist_id": "1", "verdict": None, "execution_status": "prediction_failure"},
            {"checklist_id": "2", "verdict": "Unclear", "evidence_status": "analysis_unavailable"},
            {"checklist_id": "3", "verdict": "Unclear", "execution_status": "success"},
        ]
        summary = output_node({"findings": findings, "consistency": {"status": "checked"}})["report"]["summary"]
        self.assertEqual(summary["prediction_failure"], 2)
        self.assertEqual(summary["unclear"], 1)
        self.assertEqual(summary["total"], 3)
        self.assertTrue(summary["requires_review"])

    def test_applicability_exception_is_not_semantic_uncertainty(self):
        task = task_for()
        state = {"checklist": [task["item"]], "chunks": task["chunks"], "transcript": task["transcript"]}
        model = Mock()
        model.with_structured_output.return_value.invoke.side_effect = RuntimeError()
        with patch.dict(os.environ, {"CLEARCALL_SKIP_APPLICABILITY": "false"}), \
             patch.object(applicability, "get_model", return_value=model):
            task["applicability"] = applicability.applicability_node(state)["applicability"]["fees"]
        finding = analysis._run_analysis(task)["findings"][0]
        self.assertEqual(finding["evidence_status"], "applicability_unavailable")
        self.assertIsNone(finding["verdict"])

    def test_invalid_analysis_options_rejected(self):
        for value in [0, 6, True, 1.5]:
            with self.assertRaises(ValueError):
                reliability.analysis_options({"analysis_max_attempts": value})
        with self.assertRaises(ValueError):
            reliability.analysis_options({"context_mode": "typo"})


if __name__ == "__main__":
    unittest.main()
