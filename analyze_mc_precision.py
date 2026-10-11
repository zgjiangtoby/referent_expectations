"""Estimate Monte Carlo precision from saved paired risk draws."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binom


MODELS = (("Qwen3-8B", "qwen", "Qwen"), ("EuroLLM-1.7B-Instruct", "euro", "Euro"))
LANGUAGES = {"ca": "Catalan", "cs": "Czech", "es": "Spanish"}
COUNTS = {"fixed": 2000, "refit": 100}
COMPARISONS = {
    "original": (("HPC-HP", "HPC", "HP"),),
    "local": (("HPL-HP", "HPL", "HP"), ("HPLC-HPL", "HPLC", "HPL")),
    "calibrated": (("calibrated_context", "cal_HPLC", "cal_HPL"),
                   ("calibrated_vs_raw", "cal_HPLC", "raw_HPLC")),
}
ATOL = 1e-10


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def expected_slots():
    return [(model, language, procedure, name)
            for model, _, _ in MODELS for language in LANGUAGES
            for procedure, comparisons in COMPARISONS.items()
            for name, _, _ in comparisons]


def key(row):
    return tuple(row[name] for name in ("model", "language", "procedure", "comparison"))


def validate_descriptor(row):
    model, language, procedure, comparison = key(row)
    stems = {name: stem for name, stem, _ in MODELS}
    assert model in stems and language in LANGUAGES and procedure in COMPARISONS
    heads = [(new, old) for name, new, old in COMPARISONS[procedure] if name == comparison]
    assert len(heads) == 1
    calibrated = procedure == "calibrated"
    expected = dict(new_head=heads[0][0], old_head=heads[0][1],
        family="calibrated_form_both_models" if calibrated else procedure + "_" + stems[model],
        family_size={"original": 3, "local": 6, "calibrated": 12}[procedure],
        refit_scope="temperature_and_heads" if calibrated else "heads",
        draw_population="all_eligible_opportunity_documents" if calibrated else "zero_np_documents")
    for field, value in expected.items():
        assert row[field] == value, (field, row[field], value)


def close(actual, expected):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all()
    assert np.max(np.abs(a - b), initial=0) < ATOL, (actual, expected)


def validate_draws(values, row, kind):
    assert values.shape == (COUNTS[kind],) and np.isfinite(values).all()
    close(np.quantile(values, [.025, .975]), row[kind + "_95_interval"])
    recorded_sd = row[kind + "_standard_error"]
    if recorded_sd is not None:
        close(np.std(values, ddof=1), recorded_sd)


def collect(args):
    root = args.work_dir
    family_path = root / "comparison_calibrated_reference/summary.json"
    family = read_json(family_path)
    assert family["complete"] and family["form_inference_complete"]
    assert len(family["form_comparisons"]) == 12
    family_lookup = {(r["model"], r["language"], r["name"]): r
                     for r in family["form_comparisons"]}
    assert len(family_lookup) == 12
    sources = {str(family_path.relative_to(root)): sha256(family_path)}
    descriptors, frames = [], []
    for model, stem, _ in MODELS:
        locations = {
            "original": root / ("analysis_" + stem + "_contexttokens"),
            "local": root / ("analysis_" + stem + "_local_history"),
            "calibrated": root / ("analysis_" + stem + "_calibrated_reference"),
        }
        summaries = {}
        for procedure, location in locations.items():
            path = location / "summary.json"
            summaries[procedure] = read_json(path)
            assert summaries[procedure]["completed" if procedure == "original" else "complete"]
            sources[str(path.relative_to(root))] = sha256(path)
        assert summaries["original"]["source_scores"] == summaries["local"]["source_scores"]
        assert summaries["calibrated"]["model"] == model
        for language in LANGUAGES:
            for procedure, comparisons in COMPARISONS.items():
                summary = summaries[procedure]
                result = summary["results"][language]
                suffix = {"original": language + "_bootstrap.npz",
                          "local": language + "_private_draws.npz",
                          "calibrated": "private_" + language + "_draws.npz"}[procedure]
                path = locations[procedure] / suffix
                sources[str(path.relative_to(root))] = sha256(path)
                with np.load(path, allow_pickle=False) as archive:
                    for name, new, old in comparisons:
                        if procedure == "original":
                            rows = [r for r in result["comparisons"] if r["comparison"] == name]
                        elif procedure == "local":
                            rows = [r for r in summary["comparisons"]
                                    if r["language"] == language and r["comparison"] == name]
                        else:
                            rows = [r for r in result["form_comparisons"] if r["name"] == name]
                        assert len(rows) == 1
                        row = rows[0]
                        close(result["metrics"][new]["nll"] - result["metrics"][old]["nll"], row["delta_nll"])
                        calibrated = procedure == "calibrated"
                        if calibrated:
                            combined = family_lookup[(model, language, name)]
                            for field in ("delta_nll", "fixed_pipeline_se", "joint_refit_se", "maximum_se", "normal_p_before_Holm",
                                          "fixed_pipeline_95_interval", "joint_refit_95_interval"):
                                close(row[field], combined[field])
                            adjusted = combined["holm_p_twelve"]
                            family_id, family_size = "calibrated_form_both_models", 12
                            normal_p = row["normal_p_before_Holm"]
                        else:
                            field = "holm_p_three_languages" if procedure == "original" else "holm_p_six_local_history"
                            adjusted = row[field]
                            family_size = 3 if procedure == "original" else 6
                            family_id = procedure + "_" + stem
                            normal_p = row["normal_approximation_p"]
                        descriptor = dict(model=model, language=language, procedure=procedure, comparison=name,
                            new_head=new, old_head=old, delta_nll=row["delta_nll"],
                            fixed_95_interval=row["fixed_pipeline_95_interval" if calibrated else "fixed_prediction_95_interval"],
                            refit_95_interval=row["joint_refit_95_interval" if calibrated else "paired_refit_95_interval"],
                            fixed_standard_error=row["fixed_pipeline_se"] if calibrated else None,
                            refit_standard_error=row["joint_refit_se" if calibrated else "bootstrap_standard_error"],
                            family=family_id, family_size=family_size, archived_normal_p=normal_p,
                            archived_holm_p=adjusted,
                            refit_scope="temperature_and_heads" if calibrated else "heads",
                            draw_population="all_eligible_opportunity_documents" if calibrated else "zero_np_documents")
                        validate_descriptor(descriptor)
                        descriptors.append(descriptor)
                        for kind, count in COUNTS.items():
                            prefix = kind + ("__" if calibrated else "_")
                            new_risks, old_risks = archive[prefix + new], archive[prefix + old]
                            for risks in (new_risks, old_risks):
                                assert risks.shape == (count,) and np.isfinite(risks).all()
                            values = new_risks - old_risks
                            validate_draws(values, descriptor, kind)
                            frames.append(pd.DataFrame({**{field: descriptor[field] for field in
                                ("model", "language", "procedure", "comparison")},
                                "draw_kind": kind, "replicate": np.arange(count), "delta_nll": values}))
    assert [key(r) for r in descriptors] == expected_slots()
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "contrasts.json", {"schema": "referent-risk-draws/1", "contrasts": descriptors,
        "fixed_draws": 2000, "refit_draws": 100, "new_resamples_or_fits": False})
    pd.concat(frames, ignore_index=True).to_csv(args.output / "paired_risk_differences.csv", index=False, float_format="%.17g")
    write_json(args.receipt, {"complete": True, "aggregate_only": True, "empirical_fits_or_resamples": False,
        "contrast_count": len(descriptors), "draw_rows": 63000, "source_sha256": sources,
        "public_sha256": {name: sha256(args.output / name) for name in ("contrasts.json", "paired_risk_differences.csv")}})


def sd_precision(values):
    """Deterministic delete-one jackknife Monte Carlo SE of the sample SD."""
    values = np.asarray(values, dtype=float)
    assert values.ndim == 1 and len(values) >= 3 and np.isfinite(values).all()
    if np.all(values == values[0]):
        return dict(standard_error=0.0, standard_error_mcse=0.0,
                    relative_standard_error_mcse=None)
    n = len(values)
    centered = values - values.mean()
    sum_x, sum_x2 = centered.sum(), np.dot(centered, centered)
    residual_ss = sum_x2 - centered ** 2 - (sum_x - centered) ** 2 / (n - 1)
    roundoff = 64 * np.finfo(float).eps * max(sum_x2, np.finfo(float).tiny)
    assert np.min(residual_ss) >= -roundoff
    leave_one_sd = np.sqrt(np.maximum(residual_ss, 0) / (n - 2))
    mcse = np.sqrt((n - 1) / n * np.sum((leave_one_sd - leave_one_sd.mean()) ** 2))
    sd = np.std(values, ddof=1)
    return dict(standard_error=float(sd), standard_error_mcse=float(mcse),
                relative_standard_error_mcse=float(mcse / sd) if sd > 0 else None)


def endpoint_precision(values, p):
    ordered = np.sort(np.asarray(values, dtype=float))
    n = len(ordered)
    assert n >= 3 and np.isfinite(ordered).all() and p in (.025, .975)
    lower_rank = int(binom.ppf(.025, n, p))
    upper_rank = int(binom.ppf(.975, n, p)) + 1
    lower = None if lower_rank == 0 else float(ordered[lower_rank - 1])
    upper = None if upper_rank == n + 1 else float(ordered[upper_rank - 1])
    return dict(estimate=float(np.quantile(ordered, p)), lower_rank=lower_rank, upper_rank=upper_rank,
                lower=lower, upper=upper, lower_unbounded=lower is None, upper_unbounded=upper is None,
                width=None if lower is None or upper is None else upper - lower,
                continuous_distribution_coverage=float(binom.cdf(upper_rank - 1, n, p) - binom.cdf(lower_rank - 1, n, p)))


def report(args):
    descriptor_path = args.input / "contrasts.json"
    draws_path = args.input / "paired_risk_differences.csv"
    archive = read_json(descriptor_path)
    assert archive["schema"] == "referent-risk-draws/1"
    descriptors = archive["contrasts"]
    assert [key(r) for r in descriptors] == expected_slots()
    for descriptor in descriptors:
        validate_descriptor(descriptor)
    data = pd.read_csv(draws_path, float_precision="round_trip")
    assert list(data.columns) == ["model", "language", "procedure", "comparison", "draw_kind", "replicate", "delta_nll"]
    assert len(data) == 63000 and np.isfinite(data["delta_nll"]).all()
    groups = data.groupby(["model", "language", "procedure", "comparison", "draw_kind"], sort=False)
    assert set(groups.groups) == {k + (kind,) for k in expected_slots() for kind in COUNTS}
    records = []
    for descriptor in descriptors:
        for kind, count in COUNTS.items():
            group = groups.get_group(key(descriptor) + (kind,))
            assert len(group) == count and np.array_equal(group["replicate"].to_numpy(), np.arange(count))
            values = group["delta_nll"].to_numpy()
            validate_draws(values, descriptor, kind)
            row = {field: descriptor[field] for field in ("model", "language", "procedure", "comparison", "family", "family_size")}
            row.update(draw_kind=kind, draws=count, **sd_precision(values))
            for label, p in (("lower_endpoint", .025), ("upper_endpoint", .975)):
                row.update({label + "_" + field: value for field, value in endpoint_precision(values, p).items()})
            records.append(row)
    assert len(records) == 60
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(records).to_csv(args.output / "mc_precision.csv", index=False, float_format="%.17g")
    write_json(args.output / "mc_precision.json", {"complete": True, "aggregate_only": True,
        "empirical_fits_or_resamples": False, "new_p_values": False, "comparisons": 30, "precision_records": records,
        "method": "delete-one jackknife SD MCSE; pointwise binomial order-statistic quantile sets",
        "input_sha256": {p.name: sha256(p) for p in (descriptor_path, draws_path)}})
    print(json.dumps({"complete": True, "rows": 60, "new_fits_or_draws": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    collect_parser = sub.add_parser("collect")
    collect_parser.add_argument("--work-dir", type=Path, required=True)
    collect_parser.add_argument("--output", type=Path, required=True)
    collect_parser.add_argument("--receipt", type=Path, required=True)
    report_parser = sub.add_parser("report")
    report_parser.add_argument("--input", type=Path, required=True)
    report_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    (collect if args.command == "collect" else report)(args)


if __name__ == "__main__":
    main()
