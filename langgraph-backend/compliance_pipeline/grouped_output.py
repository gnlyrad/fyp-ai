"""Retain typed sibling answers from one partially malformed provider response."""
from collections import Counter
import logging

from pydantic import ValidationError

from compliance_pipeline.schemas import CheckBatchOutput, GroupedItemAnswer

logger = logging.getLogger(__name__)


def recover_grouped_output(name, payload, schema):
    """No status inference or ID repair: validate each unique row independently.

    The caller still validates expected IDs, sub-criteria and evidence. Returning
    missing rows lets the existing production recovery handle only those items.
    """
    if schema is not CheckBatchOutput or name != schema.__name__ or not isinstance(payload, dict):
        return None
    rows = payload.get("answers")
    if not isinstance(rows, list):
        return None
    counts = Counter(row.get("item_id") for row in rows
                     if isinstance(row, dict) and isinstance(row.get("item_id"), str))
    valid = []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("item_id"), str):
            continue
        if counts[row["item_id"]] != 1:
            continue
        try:
            valid.append(GroupedItemAnswer.model_validate(row))
        except ValidationError:
            continue
    if not valid:
        return None
    logger.warning("check_batch partial parsing: retained=%d excluded=%d", len(valid), len(rows) - len(valid))
    return CheckBatchOutput(answers=valid)
