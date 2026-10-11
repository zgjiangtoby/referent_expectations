"""Count target history hidden by the fixed sentence boundary; no LLM scores used."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SOURCE_ROWS = {"ca": 12580, "es": 14513, "cs": 12139}
EXPECTED_PROBE_ROWS = {"ca": 8490, "es": 9877, "cs": 7416}


def read_rows(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def mention_count(log_count):
    value = float(np.expm1(log_count))
    rounded = int(round(value))
    assert rounded >= 1 and abs(value - rounded) < 1e-8
    return rounded


def key(row):
    return row["split"], row["doc_id"], row["sent_id"], str(row["token_id"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-items", type=Path, required=True)
    parser.add_argument("--source-rows-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    items = read_rows(args.probe_items)
    values, provenance, examples = [], [], []
    for language in ("ca", "es", "cs"):
        source_path = args.source_rows_root / f"{language}_rows.jsonl"
        source = read_rows(source_path)
        assert len(source) == EXPECTED_SOURCE_ROWS[language]
        by_key = {key(row): row for row in source}
        assert len(by_key) == len(source)
        selected = [row for row in items if row["language"] == language]
        assert len(selected) == EXPECTED_PROBE_ROWS[language]
        for row in selected:
            previous = by_key[key(row)]
            assert previous["entity_id"] == row["gold_entity"] and previous["form"] == row["gold_form"]
            before_sentence = mention_count(row["log_prior_mentions"])
            before_target = mention_count(previous["log_prior_mentions"])
            assert before_target >= before_sentence, row["item_id"]
            values.append({"item_id": row["item_id"], "language": language,
                           "doc_id": row["doc_id"], "form": row["gold_form"],
                           "before_sentence": before_sentence, "before_target": before_target,
                           "hidden_mentions": before_target - before_sentence,
                           "hidden_any": int(before_target > before_sentence)})
        # Deterministic display selection is independent of all LLM values.
        example = next((row for row in selected if row["gold_form"] == "null" and
                        3 <= row["candidate_count"] <= 6), None)
        if example is not None:
            examples.append({"language": language, "selection": "first source-order zero subject with 3--6 candidates",
                             "item": example,
                             "original_target_boundary_source_row": by_key[key(example)]})
        provenance.append({"language": language, "path": str(source_path), "sha256": file_hash(source_path),
                           "source_rows": len(source), "probe_rows": len(selected)})
    frame = pd.DataFrame(values)
    aggregates = []
    for (language, form), group in frame.groupby(["language", "form"], sort=True):
        counts = group.groupby("doc_id").size()
        weights = 1 / group.doc_id.map(counts).to_numpy(dtype=float)
        aggregates.append({"language": language, "form": form, "opportunities": len(group),
            "documents": group.doc_id.nunique(), "opportunities_with_hidden_mentions": int(group.hidden_any.sum()),
            "opportunity_fraction_hidden": float(group.hidden_any.mean()),
            "document_equal_fraction_hidden": float(np.average(group.hidden_any, weights=weights)),
            "document_equal_mean_hidden_mentions": float(np.average(group.hidden_mentions, weights=weights)),
            "max_hidden_mentions": int(group.hidden_mentions.max())})
    args.output.mkdir(parents=True)
    frame.to_csv(args.output / "private_boundary_records.csv", index=False)
    (args.output / "private_source_examples.json").write_text(json.dumps(examples, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pd.DataFrame(aggregates).to_csv(args.output / "boundary_by_form.csv", index=False)
    summary = {"rows": len(frame), "aggregates": aggregates,
        "probe_items": str(args.probe_items), "probe_items_sha256": file_hash(args.probe_items),
        "sources": provenance,
        "definition": "existing target-entity mentions completed before the annotated target start but not before its sentence boundary",
        "weighting": "document-equal within each language/form; opportunity counts also supplied",
        "scope": "descriptive source-boundary accounting; target-start positions differ with annotated realization; no model refit, prompt change, or causal contrast",
        "llm_scores_used": False, "complete": True}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "rows": len(frame), "language_form_cells": len(aggregates),
                      "llm_scores_used": False}), flush=True)


if __name__ == "__main__":
    main()
