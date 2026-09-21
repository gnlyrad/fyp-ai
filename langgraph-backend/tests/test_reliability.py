"""Regression cases from the V3-V5 review. All model/AWS calls are mocked."""
import json
import os
import unittest
from unittest.mock import Mock, patch

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from compliance_pipeline.nodes import analysis, applicability, classification, consistency, guidance, ingestion, retrieval
from compliance_pipeline.nodes.ingestion import ingestion_node, chunk_text
from compliance_pipeline.nodes.output import output_node


def task_for(question_type="deterministic", assessment="presence"):
    state = ingestion_node({
        "audio_segments": [
            {"transcript": "There is a one percent fee.", "speaker_label": "spk_0", "start_time": "0", "end_time": "4"},
            {"transcript": "I understand.", "speaker_label": "spk_1", "start_time": "5", "end_time": "6"},
        ], "speaker_roles": {"spk_0": "RM", "spk_1": "Customer"},
    })
    return {"item": {"id": "fees", "question": "Did the RM disclose fees?", "subject_role": "RM",
                     "question_type": question_type, "assessment": assessment,
                     "pass_condition": "Fees were disclosed."},
            "retrieved": [state["chunks"][0]], "chunks": state["chunks"],
            "transcript": state["transcript"], "guidance": [],
            "applicability": {"status": "applicable", "reason": "Advisory context established."},
            "retrieval_diagnostics": {"method": "semantic"}}


def judgement(verdict="Pass", basis="direct", evidence=None):
    if evidence is None:
        evidence = [{"chunk_id": "chunk-00", "quote": "There is a one percent fee.", "label": "compliant"}]
    return analysis.Judgement(verdict=verdict, basis=basis, reason="Test decision.", evidence=evidence)


def auditor(label="compliant", relation="supports", count=1):
    model = Mock()
    model.with_structured_output.return_value.invoke.return_value = analysis.EvidenceAudit(items=[
        {"index": i, "relation": relation, "label": label, "reason": "Checked relevance."}
        for i in range(count)])
    return model


class IngestionAndTimingTests(unittest.TestCase):
    def test_chunk_ids_start_at_zero_for_both_inputs(self):
        self.assertEqual(chunk_text("Hello.")[0]["chunk_id"], "chunk-00")
        task = task_for()
        self.assertEqual([c["chunk_id"] for c in task["chunks"]], ["chunk-00", "chunk-01"])

    def test_speaker_order_never_assigns_role_only_content_can(self):
        # Content too sparse to support either role: the model must say
        # "uncertain" rather than guessing from which speaker went first.
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = ingestion.SpeakerRoleInference(
            speakers=[{"speaker_label": "spk_0", "role": "uncertain", "reason": "Too little content to tell."}])
        with patch.object(ingestion, "get_model", return_value=model):
            state = ingestion_node({"audio_segments": [
                {"transcript": "Hello.", "speaker_label": "spk_0", "start_time": "0", "end_time": "1"}]})
        self.assertNotIn("speaker_role", state["chunks"][0])
        self.assertIn("role unverified", retrieval.build_context(state["chunks"]))
        self.assertTrue(state["input_warnings"])

    def test_llm_infers_roles_from_content_when_caller_omits_speaker_roles(self):
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = ingestion.SpeakerRoleInference(speakers=[
            {"speaker_label": "spk_0", "role": "RM", "reason": "Recommends the note and discloses the fee."},
            {"speaker_label": "spk_1", "role": "Customer", "reason": "Asks a clarifying question and agrees."},
        ])
        with patch.object(ingestion, "get_model", return_value=model):
            state = ingestion_node({"audio_segments": [
                {"transcript": "Here is the product we recommend, with a one percent fee.",
                 "speaker_label": "spk_0", "start_time": "0", "end_time": "3"},
                {"transcript": "Sounds good, let's proceed.", "speaker_label": "spk_1",
                 "start_time": "3", "end_time": "4"},
            ]})
        self.assertEqual(state["chunks"][0]["speaker_role"], "RM")
        self.assertEqual(state["chunks"][0]["speaker_role_source"], "llm_inferred")
        self.assertEqual(state["chunks"][1]["speaker_role"], "Customer")
        self.assertTrue(any("inferred by a language model" in w for w in state["input_warnings"]))

    def test_caller_supplied_roles_are_not_overridden_by_inference(self):
        model = Mock()
        with patch.object(ingestion, "get_model", return_value=model):
            state = ingestion_node({"audio_segments": [
                {"transcript": "Hello.", "speaker_label": "spk_0", "start_time": "0", "end_time": "1"}],
                "speaker_roles": {"spk_0": "Customer"}})
        model.assert_not_called()
        self.assertEqual(state["chunks"][0]["speaker_role"], "Customer")
        self.assertEqual(state["chunks"][0]["speaker_role_source"], "caller")

    def test_role_inference_failure_is_visible_and_does_not_block_ingestion(self):
        with patch.object(ingestion, "get_model", side_effect=RuntimeError("offline")):
            state = ingestion_node({"audio_segments": [
                {"transcript": "Here is the product.", "speaker_label": "spk_0", "start_time": "0", "end_time": "2"}]})
        self.assertNotIn("speaker_role", state["chunks"][0])
        self.assertTrue(any("inference unavailable" in w for w in state["input_warnings"]))

    def test_word_timings_map_quotes_to_words_not_whole_segment(self):
        results = {"audio_segments": [{"id": 0, "transcript": "Hello there.", "speaker_label": "spk_0",
                   "start_time": "0", "end_time": "4", "items": [10, 11, 12]}], "items": [
            {"id": 10, "type": "pronunciation", "alternatives": [{"content": "Hello"}], "start_time": "0.4", "end_time": "0.8"},
            {"id": 11, "type": "pronunciation", "alternatives": [{"content": "there"}], "start_time": "1.2", "end_time": "1.8"},
            {"id": 12, "type": "punctuation", "alternatives": [{"content": "."}]},
        ]}
        state = ingestion_node({"results": results, "speaker_roles": {"spk_0": "RM"}})
        span = analysis.locate("there.", state["chunks"], state["transcript"], "test", "chunk-00")
        self.assertEqual((span["start_seconds"], span["end_seconds"], span["timing_precision"]), (1.2, 1.8, "word"))
        results["items"][1]["alternatives"][0]["content"] = "elsewhere"
        state = ingestion_node({"results": results, "speaker_roles": {"spk_0": "RM"}})
        span = analysis.locate("there.", state["chunks"], state["transcript"], "test")
        self.assertEqual(span["timing_precision"], "segment")

    def test_invalid_timestamps_and_unknown_role_keys_are_rejected(self):
        for start, end in [("nan", "1"), ("2", "1"), ("0", "inf")]:
            with self.subTest(start=start), self.assertRaises(ValueError):
                ingestion_node({"audio_segments": [{"transcript": "Hello.", "start_time": start, "end_time": end}]})
        with self.assertRaises(ValueError):
            ingestion_node({"transcript": "Hello.", "speaker_roles": {"spk_0": "RM"}})


class RetrievalTests(unittest.TestCase):
    def test_q12_q19_normalised_fallback(self):
        chunks = chunk_text("Your objective is capital preservation. You can cancel within the cooling off period.", 1)
        self.assertEqual(retrieval.retrieve_lexical("investment objectives recommendation", chunks)[0]["chunk_id"], "chunk-00")
        for query in ("cancellation cooling-off rights", "cancellation cooling–off rights"):
            self.assertEqual(retrieval.retrieve_lexical(query, chunks)[0]["chunk_id"], "chunk-01")

    def test_query_embedding_failure_is_visible(self):
        diagnostic = {}
        with patch.object(retrieval, "embedding_client") as client:
            client.return_value.embed_query.side_effect = RuntimeError("offline")
            result = retrieval.retrieve("fee", task_for()["chunks"], [[1], [1]], diagnostics=diagnostic)
        self.assertTrue(result)
        self.assertEqual(diagnostic["method"], "lexical_fallback")
        self.assertIn("query_embedding_failed", diagnostic["fallback_reason"])

    def test_non_applicable_skips_embeddings(self):
        task = task_for()
        state = {"chunks": task["chunks"], "checklist": [task["item"]],
                 "applicability": {"fees": {"status": "not_applicable"}}}
        with patch.object(retrieval, "embed_chunks") as embed:
            result = retrieval.retrieval_node(state)
        embed.assert_not_called()
        self.assertEqual(result["retrieval_diagnostics"]["fees"]["method"], "skipped")


class ClassificationAndApplicabilityTests(unittest.TestCase):
    def test_prohibitions_are_classified_as_exhaustive_by_the_model(self):
        # No caller-supplied metadata: the LLM (not regex) must recognise these as
        # prohibited-conduct questions and classify them accordingly.
        items = [{"id": "pressure", "question": "Did the RM apply time pressure or urgency?"},
                 {"id": "guarantees", "question": "Did the RM describe any return as guaranteed?"}]
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = classification.ChecklistClassification(
            items=[
                {"id": "pressure", "question_type": "exhaustive", "assessment": "prohibition",
                 "pass_condition": "No pressure selling or undue urgency occurred anywhere in the call.",
                 "rationale": "Prohibited conduct requires whole-call coverage.",
                 "evidence_query": "time pressure urgency", "subject_role": "RM"},
                {"id": "guarantees", "question_type": "exhaustive", "assessment": "prohibition",
                 "pass_condition": "No unqualified guarantee of returns or outcomes occurred.",
                 "rationale": "Prohibited conduct requires whole-call coverage.",
                 "evidence_query": "return guarantee assure risk-free", "subject_role": "RM"},
            ])
        with patch.object(classification, "get_model", return_value=model):
            output = classification.classification_node({"checklist": items})
        model.with_structured_output.return_value.invoke.assert_called_once()
        for item in output["checklist"]:
            self.assertEqual(item["question_type"], "exhaustive")
            self.assertEqual(item["assessment"], "prohibition")
            self.assertTrue(item["pass_condition"].startswith("No "))
            self.assertEqual(item["classification_status"], "validated")

    def test_unverifiable_applicability_is_uncertain(self):
        task = task_for()
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = applicability.ApplicabilityReview(items=[
            {"id": "fees", "status": "not_applicable", "reason": "Different conversation.", "context_quotes": ["Invented text."]}])
        with patch.object(applicability, "get_model", return_value=model):
            state = applicability.applicability_node({"chunks": task["chunks"], "transcript": task["transcript"], "checklist": [task["item"]]})
        self.assertEqual(state["applicability"]["fees"]["status"], "uncertain")

    def test_applicability_failure_does_not_become_na(self):
        task = task_for()
        with patch.object(applicability, "get_model", side_effect=RuntimeError("offline")):
            state = applicability.applicability_node({"chunks": task["chunks"], "transcript": task["transcript"], "checklist": [task["item"]]})
        self.assertEqual(state["applicability"]["fees"]["status"], "uncertain")


class AnalysisTests(unittest.TestCase):
    def test_not_applicable_and_unknown_roles_do_not_call_judge(self):
        task = task_for()
        task["applicability"] = {"status": "not_applicable", "reason": "Unrelated conversation."}
        with patch.object(analysis, "judge_model") as judge:
            self.assertEqual(analysis._run_analysis(task)["raw_findings"][0]["verdict"], "N/A")
        judge.assert_not_called()
        task["applicability"]["status"] = "applicable"
        task["chunks"][0].pop("speaker_role")
        with patch.object(analysis, "judge_model") as judge:
            finding = analysis._run_analysis(task)["raw_findings"][0]
        judge.assert_not_called()
        self.assertEqual(finding["evidence_status"], "speaker_role_unverified")

    def test_absence_requires_full_review_and_has_no_violation_highlight(self):
        task = task_for("targeted")
        responses = [judgement("Unclear", "insufficient", []), judgement("Fail", "absence", [])]
        with patch.object(analysis, "_invoke", side_effect=responses) as invoke:
            finding = analysis._run_analysis(task)["raw_findings"][0]
        self.assertEqual(invoke.call_args_list[1].args[2], "full_transcript")
        self.assertEqual(finding["verdict"], "Fail")
        self.assertEqual(finding["spans"], [])
        self.assertEqual(len(finding["reviewed_chunk_ids"]), 2)
        self.assertNotIn("confidence", finding)

    def test_incomplete_coverage_never_asserts_absence(self):
        task = task_for("targeted")
        with patch.object(analysis, "_invoke", return_value=judgement("Fail", "absence", [])), \
             patch.object(analysis, "MAX_REVIEW_CHARS", len(retrieval.build_context(task["retrieved"])) + 1):
            finding = analysis._run_analysis(task)["raw_findings"][0]
        self.assertEqual(finding["verdict"], "Unclear")
        self.assertEqual(finding["evidence_status"], "coverage_incomplete")

    def test_real_but_irrelevant_quote_does_not_support_fail(self):
        with patch.object(analysis, "_invoke", return_value=judgement("Fail")), \
             patch.object(analysis, "get_model", return_value=auditor("context", "irrelevant")):
            finding = analysis._run_analysis(task_for())["raw_findings"][0]
        self.assertEqual(finding["verdict"], "Unclear")
        self.assertEqual(finding["spans"], [])

    def test_quote_labels_are_not_forced_to_verdict(self):
        with patch.object(analysis, "_invoke", return_value=judgement("Fail")), \
             patch.object(analysis, "get_model", return_value=auditor("compliant", "contradicts")):
            finding = analysis._run_analysis(task_for())["raw_findings"][0]
        self.assertEqual(finding["verdict"], "Unclear")
        self.assertEqual(finding["spans"][0]["label"], "compliant")

    def test_customer_quote_cannot_establish_rm_conduct(self):
        evidence = [{"chunk_id": "chunk-01", "quote": "I understand.", "label": "compliant"}]
        with patch.object(analysis, "_invoke", return_value=judgement(evidence=evidence)), \
             patch.object(analysis, "get_model", return_value=auditor()):
            finding = analysis._run_analysis(task_for())["raw_findings"][0]
        self.assertEqual(finding["verdict"], "Unclear")
        self.assertEqual(finding["spans"][0]["label"], "context")

    def test_repeated_quotes_are_deduplicated(self):
        answer = judgement()
        answer.evidence.append(answer.evidence[0])
        with patch.object(analysis, "_invoke", return_value=answer), \
             patch.object(analysis, "get_model", return_value=auditor()):
            finding = analysis._run_analysis(task_for())["raw_findings"][0]
        self.assertEqual(finding["verdict"], "Pass")
        self.assertEqual(len(finding["spans"]), 1)
        self.assertEqual(finding["evidence_status"], "direct_evidence")

    def test_changed_quote_with_dropped_negation_is_rejected(self):
        text = "The return is not guaranteed."
        chunks = chunk_text(text)
        self.assertIsNone(analysis.locate("The return is guaranteed.", chunks, text, "test"))

    def test_repeated_text_does_not_guess_which_occurrence_was_cited(self):
        text = "Yes. Yes."
        self.assertIsNone(analysis.locate("Yes.", chunk_text(text), text, "test"))

    def test_prohibition_pass_has_full_coverage_and_no_fake_green_quotes(self):
        task = task_for("exhaustive", "prohibition")
        with patch.object(analysis, "_invoke", return_value=judgement("Pass", "absence", [])):
            finding = analysis._run_analysis(task)["raw_findings"][0]
        self.assertEqual(finding["verdict"], "Pass")
        self.assertEqual(finding["spans"], [])
        self.assertEqual(finding["analysis_coverage"], "full_transcript")


class ConsistencyAndOutputTests(unittest.TestCase):
    def test_contradiction_quotes_are_verified_and_force_review(self):
        text = "Coupon does not depend on the basket. Coupon depends on the basket."
        chunks = chunk_text(text, 1)
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = consistency.ConsistencyReview(conflicts=[{
            "reason": "Conflicting coupon conditions.", "evidence": [
                {"chunk_id": c["chunk_id"], "quote": c["text"]} for c in chunks]}])
        state = {"chunks": chunks, "transcript": text, "raw_findings": [], "checklist": []}
        with patch.object(consistency, "get_model", return_value=model):
            consistency_result = consistency.consistency_node(state)
            state.update(consistency_result)
        summary = output_node(state)["summary"]
        self.assertTrue(summary["requires_review"])
        self.assertEqual(consistency_result["consistency"]["status"], "conflicts_found")
        self.assertEqual(len(consistency_result["consistency"]["conflicts"][0]["spans"]), 2)

    def test_one_unclear_is_enough_to_require_review(self):
        findings = [{"checklist_id": str(i), "verdict": "Pass"} for i in range(20)]
        findings[0]["verdict"] = "Unclear"
        summary = output_node({"raw_findings": findings, "consistency": {"status": "checked"}})["summary"]
        self.assertTrue(summary["requires_review"])

    def test_full_graph_retains_results_and_skips_irrelevant_compliance_calls(self):
        from compliance_pipeline.graph import graph
        source = {"transcripts": [{"transcript": "This is a comedy rehearsal."}]}
        model = Mock()
        model.with_structured_output.return_value.invoke.return_value = applicability.ApplicabilityReview(items=[
            {"id": "fees", "status": "not_applicable", "reason": "Comedy rehearsal, not financial advice.",
             "context_quotes": ["This is a comedy rehearsal."]}])
        with patch.object(applicability, "get_model", return_value=model), \
             patch.object(analysis, "judge_model") as judge, patch.object(retrieval, "embedding_client") as embed:
            output = graph.invoke({"results": source, "checklist": [task_for()["item"]]})
        judge.assert_not_called()
        embed.assert_not_called()
        self.assertNotIn("results", output)
        self.assertEqual(output["chunks"][0]["chunk_id"], "chunk-00")
        self.assertEqual(output["summary"]["not_applicable"], 1)
        self.assertNotIn("spans", output)

    def test_full_graph_mixed_routes_produces_one_consistency_report(self):
        from compliance_pipeline.graph import graph
        presence = task_for()["item"]
        prohibition = {"id": "pressure", "question": "Did the RM apply time pressure?"}
        app_model = Mock()
        app_model.with_structured_output.return_value.invoke.return_value = applicability.ApplicabilityReview(items=[
            {"id": item_id, "status": "applicable", "reason": "Advisory context.",
             "context_quotes": ["There is a one percent fee."]} for item_id in ["fees", "pressure"]])
        consistency_model = Mock()
        consistency_model.with_structured_output.return_value.invoke.return_value = consistency.ConsistencyReview()
        def decide(task, reviewed, coverage):
            return judgement() if task["item"]["id"] == "fees" else judgement("Pass", "absence", [])
        segments = [{"transcript": c["text"], "speaker_label": c["speaker_label"],
                     "start_time": str(c["start_seconds"]), "end_time": str(c["end_seconds"])}
                    for c in task_for()["chunks"]]
        with patch.object(applicability, "get_model", return_value=app_model), \
             patch.object(guidance, "search_regulations") as regulations, \
             patch.object(retrieval, "embed_chunks", return_value=None), \
             patch.object(analysis, "_invoke", side_effect=decide), \
             patch.object(analysis, "get_model", return_value=auditor()), \
             patch.object(consistency, "get_model", return_value=consistency_model):
            regulations.invoke.return_value = []
            result = graph.invoke({"audio_segments": segments, "checklist": [presence, prohibition],
                                   "speaker_roles": {"spk_0": "RM", "spk_1": "Customer"}})
        self.assertEqual(result["summary"]["pass"], 2)
        self.assertEqual(len(result["findings"]), 2)
        consistency_model.with_structured_output.return_value.invoke.assert_called_once()
        self.assertFalse(result["summary"]["requires_review"])


if __name__ == "__main__":
    unittest.main()
