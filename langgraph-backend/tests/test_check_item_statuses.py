"""The check prompt and answer schema offer PASS / FAIL / CANNOT_DETERMINE only."""
import pytest

from compliance_pipeline.prompts.check_item import SYSTEM
from compliance_pipeline.schemas import SubCriterionAnswer


def test_the_model_is_offered_three_statuses():
    schema_statuses = SubCriterionAnswer.model_json_schema()["properties"]["status"]["enum"]
    assert schema_statuses == ["PASS", "FAIL", "CANNOT_DETERMINE"]
    assert "NOT_APPLICABLE" not in SYSTEM


@pytest.mark.parametrize("stray", ["NOT_APPLICABLE", "not_applicable", "N/A", "n/a"])
def test_a_stray_not_applicable_becomes_cannot_determine_not_a_failure(stray):
    assert SubCriterionAnswer(id="met", status=stray).status == "CANNOT_DETERMINE"


def test_real_statuses_are_unchanged():
    for status in ("PASS", "FAIL", "CANNOT_DETERMINE"):
        assert SubCriterionAnswer(id="met", status=status).status == status
