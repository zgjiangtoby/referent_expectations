"""Combine the two twelve-comparison form and recovery families."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_calibrated_reference import EXPECTED, FORM_COMPARISONS, MODELS, RECOVERY_COMPARISONS, summarize_difference
from entity_analysis import holm
from diagnose_information_boundary import file_hash


def adjust_family(rows, comparisons):
    expected = {(model, language, comparison) for model in MODELS for language in EXPECTED for comparison in comparisons}
    assert len(rows) == 12 and {(r["model"], r["language"], r["name"]) for r in rows} == expected
    inputs = [1. if r["normal_p_before_Holm"] is None else r["normal_p_before_Holm"] for r in rows]
    for row, value, adjusted in zip(rows, inputs, holm(inputs)):
        row["holm_family_input"] = value
        row["holm_p_twelve"] = None if row["normal_p_before_Holm"] is None else float(adjusted)
    return all(r["normal_p_before_Holm"] is not None for r in rows)


def same_summary(actual, computed):
    for key, value in computed.items():
        if value is None or isinstance(value, str):
            assert actual[key] == value
        else:
            assert np.allclose(actual[key], value, atol=1e-12, rtol=1e-12), key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--euro", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    paths = dict(zip(MODELS, (args.qwen, args.euro)))
    summaries = {model: json.loads((path / "summary.json").read_text()) for model, path in paths.items()}
    for model, summary in summaries.items():
        assert summary["model"] == model and summary["complete"] and summary["post_result_sensitivity"]
        assert summary["original_primary_replaced"] is False and set(summary["results"]) == set(EXPECTED)
    assert len({s["readme_sha256"] for s in summaries.values()}) == 1
    form, recovery, sources = [], [], []
    for language in EXPECTED:
        provenance = [json.loads((path / f"private_{language}_draw_provenance.json").read_text()) for path in paths.values()]
        assert provenance[0] == provenance[1]
        expected = EXPECTED[language]
        p = provenance[0]
        assert p["seed"] == 20261009 and len(p["multiplicity_sha256"]) == 2000
        assert len(p["opportunity_item_ids"]) == len(p["folds"]) == expected[0]
        assert len(p["sorted_documents"]) == expected[1] and len(p["form_item_ids"]) == expected[2]
        assert len(set(p["opportunity_item_ids"])) == expected[0] and len(set(p["form_item_ids"])) == expected[2]
        assert set(p["folds"]) == set(range(5))
        for model, path in paths.items():
            result = summaries[model]["results"][language]
            assert (result["opportunities"], result["calibration_documents"], result["form_rows"], result["form_documents"]) == expected
            assert result["fixed_draws"] == 2000 and result["refit_draws"] == 100
            assert len(result["temperature_fits"]) == 5 and {r["fold"] for r in result["temperature_fits"]} == set(range(5))
            assert all(0 <= r["inverse_temperature"] <= 5 for r in result["temperature_fits"])
            selected = [r for r in result["form_comparisons"] if r["followup_family"]]
            assert len(selected) == 2 and {r["name"] for r in selected} == set(FORM_COMPARISONS)
            assert len(result["recovery_comparisons"]) == 2 and {r["name"] for r in result["recovery_comparisons"]} == set(RECOVERY_COMPARISONS)
            with np.load(path / f"private_{language}_draws.npz") as draws:
                assert draws["temperatures"].shape == (100, 5) and draws["full_temperatures"].shape == (100,)
                assert np.isfinite(draws["temperatures"]).all() and ((draws["temperatures"] >= 0) & (draws["temperatures"] <= 5)).all()
                for source in selected:
                    new, old = FORM_COMPARISONS[source["name"]]
                    assert source["new"] == new and source["old"] == old
                    point = result["metrics"][new]["nll"] - result["metrics"][old]["nll"]
                    computed = summarize_difference(point, draws["fixed__" + new] - draws["fixed__" + old],
                                                     draws["refit__" + new] - draws["refit__" + old])
                    same_summary(source, computed)
                    form.append(dict(model=model, language=language, **source))
                for source in result["recovery_comparisons"]:
                    key = source["name"]
                    computed = summarize_difference(result["recovery"][key]["mean"], draws["construct_fixed__" + key], draws["construct_refit__" + key])
                    same_summary(source, computed)
                    recovery.append(dict(model=model, language=language, **source))
            sources.append({"model": model, "language": language,
                "summary_sha256": file_hash(path / "summary.json"),
                "draws_sha256": file_hash(path / f"private_{language}_draws.npz"),
                "provenance_sha256": file_hash(path / f"private_{language}_draw_provenance.json")})
    complete_form = adjust_family(form, FORM_COMPARISONS)
    complete_recovery = adjust_family(recovery, RECOVERY_COMPARISONS)
    args.output.mkdir(parents=True)
    pd.DataFrame(form).to_csv(args.output / "form_twelve_comparisons.csv", index=False)
    pd.DataFrame(recovery).to_csv(args.output / "recovery_twelve_comparisons.csv", index=False)
    (args.output / "summary.json").write_text(json.dumps({"complete": True,
        "inference_complete": complete_form and complete_recovery,
        "form_inference_complete": complete_form, "recovery_inference_complete": complete_recovery,
        "form_comparisons": form, "recovery_comparisons": recovery, "family_sizes": {"form": 12, "recovery": 12},
        "sources": sources, "post_result_sensitivity": True, "original_primary_replaced": False,
        "scope": "fixed original folds and LLM scores; 2000 full-calibration-document draws, first 100 joint temperature/head refits",
        "p_value_scope": "separate twelve-slot form and recovery families; two-sided maximum-SE normal approximation with Holm; post-result, not preregistered confirmation"},
        indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "inference_complete": complete_form and complete_recovery,
                      "form_family": 12, "recovery_family": 12}), flush=True)


if __name__ == "__main__":
    main()
