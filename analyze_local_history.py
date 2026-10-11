"""Compare local-history and contextual features on the same document folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
from entity_analysis import contrast, grouped_evaluation, holm
from diagnose_information_boundary import EXPECTED_PROBE_ROWS, file_hash


def read_analysis_csv(path):
    """Preserve the literal zero-subject category ``null`` across CSV storage."""
    return pd.read_csv(path, float_precision="round_trip", keep_default_na=False)


def six_family(rows):
    expected = {(language, comparison) for language in ("ca", "cs", "es")
                for comparison in ("HPL-HP", "HPLC-HPL")}
    assert len(rows) == 6 and {(r["language"], r["comparison"]) for r in rows} == expected
    complete, inputs = True, []
    for row in rows:
        delta, se = row["delta_nll"], row["bootstrap_standard_error"]
        assert np.isfinite([delta, se]).all() and se >= 0
        if se <= 1e-12:
            if abs(delta) <= 1e-12:
                p, status = 1., "numerically_zero_contrast"
            else:
                p, status = None, "nonzero_contrast_with_degenerate_resampling"
                complete = False
        else:
            p, status = float(2 * norm.sf(abs(delta) / se)), "normal_approximation"
        row["normal_approximation_p"] = p
        row["inference_status"] = status
        row["family_input_p"] = 1. if p is None else p
        inputs.append(row["family_input_p"])
    for row, p in zip(rows, holm(inputs)):
        row["holm_p_six_local_history"] = None if row["normal_approximation_p"] is None else float(p)
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refits", type=int, default=100)
    args = parser.parse_args()
    assert not args.output.exists() and args.refits == 100
    original = json.loads((args.analysis / "summary.json").read_text())
    boundary = json.loads((args.boundary / "summary.json").read_text())
    assert original["completed"] and boundary["complete"] and boundary["llm_scores_used"] is False
    features = read_analysis_csv(args.analysis / "features.csv")
    history = read_analysis_csv(args.boundary / "private_boundary_records.csv")
    assert features.item_id.is_unique and history.item_id.is_unique
    assert len(features) == len(history) == sum(EXPECTED_PROBE_ROWS.values())
    assert set(features.item_id) == set(history.item_id)
    before_order = features.item_id.tolist()
    frame = features.merge(history[["item_id", "language", "doc_id", "form", "hidden_mentions"]],
                           on=["item_id", "language", "doc_id"], how="left", validate="one_to_one", sort=False)
    assert frame.item_id.tolist() == before_order and frame.gold_form.equals(frame.form)
    assert set(frame.gold_form) == {"null", "np", "pronoun"}
    assert frame.hidden_mentions.notna().all() and frame.hidden_mentions.ge(0).all()
    assert np.equal(frame.hidden_mentions, np.floor(frame.hidden_mentions)).all()
    frame["log_hidden_mentions"] = np.log1p(frame.hidden_mentions)
    baseline = ["log_distance", "log_prior_mentions", "prior_subject", "log_candidates", "log_matching_count",
                "log_unknown_count", "log_ambiguous_count", "current_plural", "aux_no_part", "has_part",
                "clause_coord", "clause_embedded", "P_surprisal", "P_entropy"]
    columns = {"HP": baseline, "HPL": baseline + ["log_hidden_mentions"],
               "HPLC": baseline + ["log_hidden_mentions", "DP_surprisal_increment", "DP_entropy_increment"]}
    previous_predictions = read_analysis_csv(args.analysis / "heldout_predictions.csv")
    assert previous_predictions.item_id.is_unique
    args.output.mkdir(parents=True)
    (args.output / "run.json").write_text(json.dumps({
        "analysis": str(args.analysis), "boundary": str(args.boundary), "refits": args.refits,
        "analysis_summary_sha256": file_hash(args.analysis / "summary.json"),
        "feature_sha256": file_hash(args.analysis / "features.csv"),
        "boundary_summary_sha256": file_hash(args.boundary / "summary.json"),
        "readme_sha256": file_hash(HERE / "README.md"),
        "columns": columns, "new_fits_opened": False,
        "timing": "after Euro outcomes and source-boundary accounting, before new local-history fits and Qwen inference",
    }, indent=2, allow_nan=False) + "\n")
    results, all_comparisons, predictions = {}, [], []
    for language, selected in frame[frame.gold_form.isin(["null", "np"])].groupby("language"):
        selected = selected.reset_index(drop=True)
        assert (len(selected), selected.doc_id.nunique()) == (
            original["results"][language]["rows"], original["results"][language]["groups"])
        result = grouped_evaluation(selected, columns, "doc_id", "fold", True, args.refits, 20261009)
        previous = previous_predictions[previous_predictions.language.eq(language)].reset_index(drop=True)
        assert selected.item_id.equals(previous.item_id)
        assert selected.fold.equals(previous.fold) and selected.y.equals(previous.y)
        differences = {"heldout_probability": float(np.max(np.abs(result["predictions"]["HP"] - previous.HP)))}
        old_draws = np.load(args.analysis / f"{language}_bootstrap.npz")
        for prefix, name in (("fixed", "fixed_draws"), ("refit", "refit_draws")):
            old = old_draws[f"{prefix}_HP"]
            new = result[name]["HP"]
            assert old.shape == new.shape
            differences[prefix + "_risk_draws"] = float(np.max(np.abs(old - new)))
        assert max(differences.values()) < 1e-7, (language, differences)
        comparisons = [{"language": language, **contrast(result, new, old)}
                       for new, old in (("HPL", "HP"), ("HPLC", "HPL"))]
        all_comparisons.extend(comparisons)
        results[language] = {k: result[k] for k in ("rows", "groups", "weighting", "metrics", "fits", "full_fits", "numerics")}
        results[language]["original_HP_replication_max_abs_difference"] = differences
        rows = selected[["item_id", "language", "doc_id", "fold", "gold_form", "y", "hidden_mentions"]].copy()
        for name, p in result["predictions"].items():
            rows[name] = p
        predictions.append(rows)
        np.savez_compressed(args.output / f"{language}_private_draws.npz",
                            **{"fixed_" + k: v for k, v in result["fixed_draws"].items()},
                            **{"refit_" + k: v for k, v in result["refit_draws"].items()})
        print(json.dumps({"language": language, "local_history_refits_completed": args.refits,
                          "original_HP_replication_max_abs_difference": differences}), flush=True)
    inference_complete = six_family(all_comparisons)
    pd.concat(predictions, ignore_index=True).to_csv(args.output / "private_heldout_predictions.csv", index=False)
    summary = {"results": results, "comparisons": all_comparisons,
        "family": "six additional local-history/conditional-context comparisons within this checkpoint",
        "source_scores": original["source_scores"], "refits": args.refits,
        "interpretation": "conditional source-corpus prediction; local-history endpoints depend on native target annotation, not a form-invariant intervention",
        "llm_prompts_changed": False, "original_primary_replaced": False,
        "inference_complete": inference_complete, "complete": True}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "inference_complete": inference_complete,
                      "comparisons": len(all_comparisons), "refits": args.refits}), flush=True)


if __name__ == "__main__":
    main()
