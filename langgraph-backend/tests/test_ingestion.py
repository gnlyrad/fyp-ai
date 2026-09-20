"""Transcribe input contract and state preservation, without model calls."""
from __future__ import annotations

import copy
import os
import unittest

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langgraph.graph import END, START, StateGraph

from compliance_pipeline.nodes.ingestion import ingestion_node
from compliance_pipeline.nodes.output import output_node
from compliance_pipeline.state import AnalyseInput, ComplianceState


class IngestionTests(unittest.TestCase):
    def setUp(self):
        self.results = {
            "transcripts": [{"transcript": "Hello. Yes, go ahead."}],
            "speaker_labels": {"speakers": 2, "segments": []},
            "items": [{"type": "pronunciation", "start_time": "0.2"}],
            "audio_segments": [
                {"id": 0, "transcript": "Hello.", "speaker_label": "spk_0",
                 "start_time": "0.2", "end_time": "1.1", "items": [0]},
                {"id": 1, "transcript": "Yes, go ahead.", "speaker_label": "spk_1",
                 "start_time": "1.5", "end_time": "2.8", "items": []},
            ],
        }
        self.checklist = [{"id": "greeting", "question": "Was a greeting given?"}]

    def test_results_preserve_metadata_and_turn_offsets(self):
        original = copy.deepcopy(self.results)
        state = ingestion_node({"results": self.results, "checklist": self.checklist})
        self.assertEqual(state["results"], original)
        self.assertEqual(self.results, original)
        self.assertEqual(state["audio_segments"], original["audio_segments"])
        self.assertEqual(state["checklist"], self.checklist)
        self.assertEqual(state["transcript"], "Hello.\n\nYes, go ahead.")
        for chunk, segment in zip(state["chunks"], original["audio_segments"]):
            self.assertEqual(chunk["speaker_label"], segment["speaker_label"])
            self.assertEqual(chunk["start_seconds"], float(segment["start_time"]))
            self.assertEqual(chunk["end_seconds"], float(segment["end_time"]))
            self.assertEqual(
                state["transcript"][chunk["start_char"]:chunk["end_char"]], chunk["text"]
            )

    def test_results_take_priority_over_legacy_inputs(self):
        state = ingestion_node({"results": self.results, "transcript": "Ignore me.",
                                "audio_segments": [{"transcript": "Ignore me too."}]})
        self.assertEqual(state["chunks"][0]["text"], "Hello.")

    def test_results_without_segments_use_transcript_without_inventing_speakers(self):
        results = {"transcripts": [{"transcript": "  Plain text.  "}], "items": []}
        state = ingestion_node({"results": results, "audio_segments": self.results["audio_segments"]})
        self.assertEqual(state["transcript"], "Plain text.")
        self.assertEqual(state["audio_segments"], [])
        self.assertNotIn("speaker_label", state["chunks"][0])
        self.assertNotIn("start_seconds", state["chunks"][0])
        self.assertEqual(state["results"], results)

    def test_empty_segments_fall_back_to_transcripts(self):
        self.results["audio_segments"] = []
        state = ingestion_node({"results": self.results})
        self.assertEqual(state["transcript"], "Hello. Yes, go ahead.")

    def test_invalid_results_do_not_silently_use_another_input(self):
        for results in (None, [], {}, {"audio_segments": {}},
                        {"transcripts": [None]}, {"transcripts": [{"transcript": " "}]}):
            with self.subTest(results=results), self.assertRaises(ValueError):
                ingestion_node({"results": results, "transcript": "Old transcript."})

    def test_legacy_inputs_still_work(self):
        text_state = ingestion_node({"transcript": "Hello. Welcome."})
        self.assertEqual(text_state["transcript"], "Hello. Welcome.")
        self.assertNotIn("results", text_state)
        segment_state = ingestion_node({"audio_segments": self.results["audio_segments"],
                                        "transcript": "Ignored."})
        self.assertEqual(segment_state["chunks"][1]["speaker_label"], "spk_1")

    def test_graph_accepts_results_and_returns_them_alongside_report(self):
        # Exercise the public input schema and actual output node without calling
        # classification, retrieval, or a paid model.
        builder = StateGraph(ComplianceState, input_schema=AnalyseInput)
        builder.add_node("ingestion", ingestion_node)
        builder.add_node("output", output_node)
        builder.add_edge(START, "ingestion")
        builder.add_edge("ingestion", "output")
        builder.add_edge("output", END)
        state = builder.compile().invoke({"results": self.results, "checklist": self.checklist})
        self.assertEqual(state["results"], self.results)
        self.assertEqual(state["audio_segments"], self.results["audio_segments"])
        self.assertEqual(len(state["chunks"]), 2)
        self.assertIn("report", state)


if __name__ == "__main__":
    unittest.main()
