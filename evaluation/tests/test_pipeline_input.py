from copy import deepcopy

from fyp_evaluation_engine.current_pipeline import prepare_input


def test_bridge_preserves_rule_polarity_without_forwarding_benchmark_annotations():
    value = {
        "results": {"audio_segments": []},
        "speaker_roles": {"speaker_0": "EMPLOYEE"},
        "checklist": [
            {"id": "TEST_O", "question": "Explain the fictional risk.",
             "requirement_type": "OBLIGATION", "expected": {"verdict": "FAIL"},
             "gold_evidence_turns": [2], "challenge_tags": ["test"]},
            {"id": "TEST_P", "question": "Do not promise fictional returns.",
             "requirement_type": "PROHIBITION", "evaluation_family": "compliance"},
            {"id": "TEST_X", "question": "What is the fictional quantity?",
             "field": "quantity", "expected": {"value": 123}},
            {"id": "TEST_LEGACY", "question": "Legacy question without metadata."},
        ],
    }
    original = deepcopy(value)
    prepared = prepare_input(value)
    assert prepared["checklist"] == [
        {"id": "TEST_O", "question": "Explain the fictional risk.", "requirement_type": "OBLIGATION"},
        {"id": "TEST_P", "question": "Do not promise fictional returns.", "requirement_type": "PROHIBITION"},
        {"id": "TEST_X", "question": "What is the fictional quantity?"},
        {"id": "TEST_LEGACY", "question": "Legacy question without metadata."},
    ]
    assert prepared["results"] == original["results"]
    assert prepared["speaker_roles"] == original["speaker_roles"]
    assert value == original
