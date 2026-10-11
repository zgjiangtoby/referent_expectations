"""Refit temperature and form heads together within document folds."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.special import logsumexp
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
import entity_analysis
from entity_analysis import binary_losses, fit_predict, read_jsonl, read_probes
from analyze_local_history import read_analysis_csv
from diagnose_information_boundary import file_hash

CONDITIONS = ("D", "P", "DP")
MODELS = {"Qwen3-8B": ("b968826d9c46dd6066d109eabc6255188de91218", 8),
          "EuroLLM-1.7B-Instruct": ("a25c7fa65fc2a644e6270b8940dbe295b51da681", 32)}
EXPECTED = {"ca": (8490, 1145, 8423, 1144), "cs": (7416, 2083, 7367, 2081),
            "es": (9877, 1202, 9766, 1202)}
HISTORY = ["log_distance", "log_prior_mentions", "prior_subject", "log_candidates", "log_matching_count",
           "log_unknown_count", "log_ambiguous_count", "current_plural", "aux_no_part", "has_part",
           "clause_coord", "clause_embedded"]
NAMES = ("raw_HPL", "raw_HPLC", "cal_HPL", "cal_HPLC")
FORM_COMPARISONS = {"calibrated_context": ("cal_HPLC", "cal_HPL"),
                    "calibrated_vs_raw": ("cal_HPLC", "raw_HPLC")}
RECOVERY_COMPARISONS = ("cal_DP_minus_P", "cal_DP_minus_uniform")


def transform(logp, mask, target, alpha):
    assert 0 <= alpha <= 5
    logits = np.where(mask, alpha * logp, -np.inf)
    logz = logsumexp(logits, axis=-1)
    probability = np.exp(logits - logz[..., None])
    gold_log = np.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
    loss = logz - alpha * gold_log
    derivative = np.sum(probability * logp, axis=-1) - gold_log
    entropy = logz - alpha * np.sum(probability * logp, axis=-1)
    denominator = np.log(mask.sum(axis=-1))
    entropy = np.divide(entropy, denominator, out=np.zeros_like(entropy), where=denominator > 0)
    assert np.isfinite(loss).all() and np.isfinite(derivative).all() and np.isfinite(entropy).all()
    assert np.allclose(probability.sum(axis=-1), 1., atol=1e-12, rtol=0.)
    return loss, derivative, entropy, probability


def fit_temperature(logp, mask, target, weights, train):
    selected = train & (weights > 0)
    assert selected.any() and np.isfinite(weights).all() and (weights >= 0).all()

    def evaluate(alpha):
        loss, derivative, _, _ = transform(logp[selected], mask[selected], target[selected], alpha)
        return (float(np.average(loss.mean(axis=1), weights=weights[selected])),
                float(np.average(derivative.mean(axis=1), weights=weights[selected])))

    low, high = evaluate(0.)[1], evaluate(5.)[1]
    assert high >= low - 1e-10
    if low >= 0:
        alpha, status = 0., "uniform_boundary"
    elif high <= 0:
        alpha, status = 5., "upper_boundary"
    else:
        alpha, status = float(brentq(lambda a: evaluate(a)[1], 0., 5., xtol=1e-12)), "interior"
    value, derivative = evaluate(alpha)
    assert derivative >= -1e-8 if alpha == 0 else derivative <= 1e-8 if alpha == 5 else abs(derivative) < 1e-8
    assert value <= min(evaluate(0.)[0], evaluate(1.)[0]) + 1e-10
    return {"inverse_temperature": alpha, "status": status, "training_nll": value,
            "training_derivative": derivative, "training_opportunities": int(selected.sum()),
            "training_weight_sum": float(weights[selected].sum())}


def summarize_difference(point, fixed, refit):
    assert len(fixed) == 2000 and len(refit) == 100
    assert np.isfinite(fixed).all() and np.isfinite(refit).all()
    fixed_se, refit_se = float(np.std(fixed, ddof=1)), float(np.std(refit, ddof=1))
    se = max(fixed_se, refit_se)
    if se <= 1e-12:
        p, status = ((1., "numerically_zero_contrast") if abs(point) <= 1e-12 else
                     (None, "nonzero_contrast_with_degenerate_resampling"))
    else:
        p, status = float(2 * norm.sf(abs(point) / se)), "normal_approximation"
    return {"delta_nll": float(point), "fixed_pipeline_95_interval": np.quantile(fixed, [.025, .975]).tolist(),
            "joint_refit_95_interval": np.quantile(refit, [.025, .975]).tolist(),
            "fixed_pipeline_se": fixed_se, "joint_refit_se": refit_se, "maximum_se": se,
            "normal_p_before_Holm": p, "inference_status": status, "fixed_draws": 2000, "refit_draws": 100}


def load_data(args):
    original = json.loads((args.reference_analysis / "summary.json").read_text())
    local = json.loads((args.local_history / "summary.json").read_text())
    assert original["completed"] and local["complete"]
    assert Path(original["source_scores"]).resolve() == Path(local["source_scores"]).resolve() == args.scores.resolve()
    metadata_path = args.scores.with_suffix(".metadata.json")
    metadata = json.loads(metadata_path.read_text())
    revision, batch = MODELS[args.model_name]
    assert metadata["status"] == "complete" and metadata["rows"] == 154698
    assert metadata["dtype"] == "float32" and metadata["gold_used"] is False
    assert Path(metadata["model_path"]).name == revision
    source = HERE / "work/probes"
    assert Path(metadata["input"]).resolve() == (source / "inputs.jsonl").resolve()
    assert metadata["manifest"]["input_sha256"] == file_hash(source / "inputs.jsonl")
    assert metadata["output_sha256"] == file_hash(args.scores)
    assert metadata["completed_shards"]
    for shard in metadata["completed_shards"]:
        job = shard["metadata"]
        assert job["status"] == "complete" and job["gold_used"] is False
        assert Path(job["model_path"]).name == revision and job["dtype"] == "float32" and job["batch_size"] == batch
    items = read_jsonl(source / "items.jsonl")
    probes = read_probes(source, args.scores)
    frame = read_analysis_csv(args.reference_analysis / "features.csv")
    history = read_analysis_csv(args.boundary / "private_boundary_records.csv")
    assert len(items) == len(frame) == len(history) == 25783
    assert frame.item_id.is_unique and history.item_id.is_unique
    assert frame.item_id.tolist() == [r["item_id"] for r in items]
    for key in ("language", "doc_id", "fold", "gold_form", "candidate_count"):
        assert frame[key].tolist() == [r[key] for r in items]
    before = frame.item_id.tolist()
    frame = frame.merge(history[["item_id", "language", "doc_id", "form", "hidden_mentions"]],
                        on=["item_id", "language", "doc_id"], how="left", sort=False, validate="one_to_one")
    assert frame.item_id.tolist() == before and frame.gold_form.equals(frame.form)
    assert set(frame.gold_form) == {"null", "np", "pronoun"}
    assert frame.hidden_mentions.notna().all() and frame.hidden_mentions.ge(0).all()
    assert np.equal(frame.hidden_mentions, np.floor(frame.hidden_mentions)).all()
    assert frame.y.tolist() == frame.gold_form.eq("np").astype(int).tolist()
    frame["log_hidden_mentions"] = np.log1p(frame.hidden_mentions)
    counts = np.array([len(r["candidates"]) for r in items])
    assert np.array_equal(counts, frame.candidate_count.to_numpy()) and (counts >= 1).all()
    logp = np.zeros((len(items), 3, int(counts.max())))
    mask = np.zeros_like(logp, dtype=bool)
    target = np.zeros((len(items), 3), dtype=int)
    for i, row in enumerate(items):
        expected_entities = {c["entity_id"] for c in row["candidates"]}
        for j, condition in enumerate(CONDITIONS):
            distribution = probes[row["item_id"], "subject", condition]["probabilities"]
            # Preserve the archived mapping's first-maximum tie convention.
            entities = list(distribution)
            assert set(entities) == expected_entities
            p = np.array([distribution[e] for e in entities])
            assert np.isfinite(p).all() and (p >= 1e-300).all() and np.isclose(p.sum(), 1., atol=1e-12, rtol=0.)
            logp[i, j, :len(p)] = np.log(p)
            mask[i, j, :len(p)] = True
            target[i, j] = entities.index(row["gold_entity"])
    loss, _, entropy, probability = transform(logp, mask, target, 1.)
    reproduction = {}
    for j, condition in enumerate(CONDITIONS):
        for name, values in (("surprisal", loss), ("entropy", entropy)):
            difference = float(np.max(np.abs(values[:, j] - frame[condition + "_" + name].to_numpy())))
            assert difference < 1e-12
            reproduction[condition + "_" + name] = difference
        assert np.array_equal(probability[:, j].argmax(axis=-1) == target[:, j], frame[condition + "_correct"].to_numpy())
    return {"frame": frame, "logp": logp, "mask": mask, "target": target, "counts": counts,
            "original": original, "local": local, "raw_measurement_reproduction": reproduction,
            "scoring_source_hashes": [j["metadata"].get("runtime_source_sha256") for j in metadata["completed_shards"]],
            "prefix_rows": metadata["manifest"]["prefix_rows"]}


def matrices(frame, selected, losses, entropy):
    base = frame.loc[selected, HISTORY].to_numpy(dtype=float)
    local = frame.loc[selected, "log_hidden_mentions"].to_numpy(dtype=float)
    raw_prior = frame.loc[selected, ["P_surprisal", "P_entropy"]].to_numpy(dtype=float)
    raw_context = frame.loc[selected, ["DP_surprisal_increment", "DP_entropy_increment"]].to_numpy(dtype=float)
    cal_prior = np.column_stack((losses[selected, 1], entropy[selected, 1]))
    cal_context = np.column_stack((losses[selected, 2] - losses[selected, 1], entropy[selected, 2] - entropy[selected, 1]))
    raw, calibrated = np.column_stack((base, raw_prior, local)), np.column_stack((base, cal_prior, local))
    return {"raw_HPL": raw, "raw_HPLC": np.column_stack((raw, raw_context)),
            "cal_HPL": calibrated, "cal_HPLC": np.column_stack((calibrated, cal_context))}


def construct_values(raw_loss, cal_loss, raw_accuracy, cal_accuracy, counts):
    values = {}
    for j, condition in enumerate(CONDITIONS):
        values.update({"raw_" + condition + "_nll": raw_loss[:, j], "cal_" + condition + "_nll": cal_loss[:, j],
                       "cal_minus_raw_" + condition: cal_loss[:, j] - raw_loss[:, j],
                       "raw_" + condition + "_accuracy": raw_accuracy[:, j], "cal_" + condition + "_accuracy": cal_accuracy[:, j]})
    values.update(cal_DP_minus_P=cal_loss[:, 2] - cal_loss[:, 1], cal_DP_minus_uniform=cal_loss[:, 2] - np.log(counts),
                  cal_D_minus_P=cal_loss[:, 0] - cal_loss[:, 1], cal_DP_minus_D=cal_loss[:, 2] - cal_loss[:, 0])
    return values


def run_language(data, language, args):
    position = np.flatnonzero(data["frame"].language.eq(language).to_numpy())
    frame = data["frame"].iloc[position].reset_index(drop=True)
    logp, mask, target, counts = [data[k][position] for k in ("logp", "mask", "target", "counts")]
    selected = np.flatnonzero(frame.gold_form.isin(["null", "np"]).to_numpy())
    groups, group_index = np.unique(frame.doc_id.astype(str), return_inverse=True)
    folds = frame.fold.to_numpy(dtype=int)
    y = frame.loc[selected, "y"].to_numpy(dtype=int)
    assert (len(frame), len(groups), len(selected), frame.loc[selected, "doc_id"].nunique()) == EXPECTED[language]
    assert frame.groupby("doc_id").fold.nunique().eq(1).all() and set(folds) == set(range(5))
    opportunity_weights = 1 / np.bincount(group_index)[group_index]
    form_weights = 1 / np.bincount(group_index[selected], minlength=len(groups))[group_index[selected]]
    raw_loss, _, raw_entropy, raw_probability = transform(logp, mask, target, 1.)
    raw_accuracy = raw_probability.argmax(axis=-1) == target
    raw_matrices = matrices(frame, selected, raw_loss, raw_entropy)
    for new, old in (("cal_HPL", "raw_HPL"), ("cal_HPLC", "raw_HPLC")):
        assert np.max(np.abs(raw_matrices[new] - raw_matrices[old])) < 1e-12
    # All original scalar fits for this cell precede its first calibrated prediction/head fit.
    original_temperatures = [dict(fold=fold, **fit_temperature(logp, mask, target, opportunity_weights, folds != fold))
                             for fold in range(5)]
    (args.output / f"{language}_temperature_freeze.json").write_text(json.dumps({
        "model": args.model_name, "language": language, "fits": original_temperatures,
        "post_result": True, "new_calibrated_evaluation_computed": False,
        "training_target": "existing referent links, not reference-form labels"}, indent=2, allow_nan=False) + "\n")
    fits, full_fits, fallbacks = [], {}, []
    numerical = {"fit_calls": 0, "maximum_gradient_per_training_weight": 0.}

    def keep(record, name, draw, fold, phase):
        numerical["fit_calls"] += 1
        numerical["maximum_gradient_per_training_weight"] = max(numerical["maximum_gradient_per_training_weight"], record["gradient_max_per_training_weight"])
        if draw is None:
            fits.append(dict(model=name, fold=fold, phase=phase, **record))
        if record["fallback"] is not None:
            fallbacks.append(dict(model=name, draw=draw, fold=fold, phase=phase, **record))

    def pipeline(multiplicity, draw):
        opportunity = opportunity_weights * multiplicity[group_index]
        head_weights = form_weights * multiplicity[group_index[selected]]
        prediction = {name: np.full(len(selected), np.nan) for name in NAMES}
        oof_loss, oof_accuracy = np.full_like(raw_loss, np.nan), np.full_like(raw_loss, np.nan)
        temperatures = []
        for fold in range(5):
            record = (original_temperatures[fold] if draw is None else
                      fit_temperature(logp, mask, target, opportunity, folds != fold))
            alpha = record["inverse_temperature"]
            temperatures.append(alpha)
            loss, _, entropy, probability = transform(logp, mask, target, alpha)
            oof_loss[folds == fold] = loss[folds == fold]
            oof_accuracy[folds == fold] = (probability.argmax(axis=-1) == target)[folds == fold]
            xs = matrices(frame, selected, loss, entropy)
            train = (folds[selected] != fold) & (head_weights > 0)
            test = (folds[selected] == fold) & (head_weights > 0)
            if not test.any():
                continue
            expanded = int(multiplicity[group_index[selected][train]].sum())
            for name in NAMES:
                p, fitted = fit_predict(xs[name], y, head_weights, train, test, expanded)
                prediction[name][test] = p
                keep(fitted, name, draw, fold, "outer_fold")
        present = head_weights > 0
        assert all(np.isfinite(p[present]).all() for p in prediction.values())
        risk = {name: float(np.average(binary_losses(y[present], p[present]), weights=head_weights[present]))
                for name, p in prediction.items()}
        full_temperature = fit_temperature(logp, mask, target, opportunity, np.ones(len(frame), dtype=bool))
        full_loss, _, full_entropy, _ = transform(logp, mask, target, full_temperature["inverse_temperature"])
        xs = matrices(frame, selected, full_loss, full_entropy)
        coefficients = {}
        names = ("raw_HPL", "raw_HPLC", "cal_HPLC") if draw is None else ("raw_HPLC", "cal_HPLC")
        for name in names:
            _, fitted = fit_predict(xs[name], y, head_weights, present, np.ones(len(y), dtype=bool),
                                    int(multiplicity[group_index[selected][present]].sum()))
            keep(fitted, name, draw, None, "full_descriptive_coefficient")
            if draw is None:
                full_fits[name] = fitted
            if name.endswith("HPLC"):
                coefficients[name] = np.r_[fitted["intercept"], fitted["coefficients_original_units"]]
        construct = construct_values(raw_loss, oof_loss, raw_accuracy, oof_accuracy, counts)
        constructed = {key: float(np.average(value, weights=opportunity)) for key, value in construct.items()}
        return prediction, risk, coefficients, construct, constructed, temperatures, full_temperature["inverse_temperature"]

    point_predictions, point, point_coef, constructed, construct_point, _, full_alpha = pipeline(np.ones(len(groups), dtype=int), None)
    archived = read_analysis_csv(args.local_history / "private_heldout_predictions.csv")
    archived = archived.loc[archived.language.eq(language)].reset_index(drop=True)
    assert frame.loc[selected, "item_id"].tolist() == archived.item_id.tolist()
    assert frame.loc[selected, "fold"].tolist() == archived.fold.tolist() and y.tolist() == archived.y.tolist()
    reproduction = {}
    for name, old_name in (("raw_HPL", "HPL"), ("raw_HPLC", "HPLC")):
        difference = float(np.max(np.abs(point_predictions[name] - archived[old_name].to_numpy())))
        assert difference < 1e-7
        reproduction[name + "_probability"] = difference
        previous = data["local"]["results"][language]["full_fits"][old_name]
        coefficient = np.r_[full_fits[name]["intercept"], full_fits[name]["coefficients_original_units"]]
        difference = float(np.max(np.abs(coefficient - np.r_[previous["intercept"], previous["coefficients_original_units"]])))
        assert difference < 1e-7
        reproduction[name + "_full_coefficient"] = difference
    losses = {name: binary_losses(y, p) for name, p in point_predictions.items()}
    fixed, refitted = {k: [] for k in NAMES}, {k: [] for k in NAMES}
    construct_fixed, construct_refit = {k: [] for k in constructed}, {k: [] for k in constructed}
    coefficient_draws = {k: [] for k in point_coef}
    temperature_draws, full_temperatures, draw_ids = [], [], []
    rng = np.random.default_rng(20261009)
    for draw in range(2000):
        multiplicity = rng.multinomial(len(groups), np.full(len(groups), 1 / len(groups)))
        draw_ids.append(hashlib.sha256(multiplicity.astype("<i8").tobytes()).hexdigest())
        for name in NAMES:
            fixed[name].append(float(np.average(losses[name], weights=form_weights * multiplicity[group_index[selected]])))
        for name, values in constructed.items():
            construct_fixed[name].append(float(np.average(values, weights=opportunity_weights * multiplicity[group_index])))
        if draw < 100:
            _, risks, coef, _, constructs, temperatures, full_temp = pipeline(multiplicity, draw)
            for name in NAMES:
                refitted[name].append(risks[name])
            for name in constructed:
                construct_refit[name].append(constructs[name])
            for name in coefficient_draws:
                coefficient_draws[name].append(coef[name])
            temperature_draws.append(temperatures)
            full_temperatures.append(full_temp)
            if (draw + 1) % 20 == 0:
                print(json.dumps({"language": language, "joint_temperature_head_refits": draw + 1}), flush=True)
    fixed, refitted = [{k: np.asarray(v) for k, v in values.items()} for values in (fixed, refitted)]
    form_comparisons = []
    for key, (new, old) in {**FORM_COMPARISONS, "raw_context": ("raw_HPLC", "raw_HPL")}.items():
        form_comparisons.append(dict(name=key, new=new, old=old, followup_family=key in FORM_COMPARISONS,
            **summarize_difference(point[new] - point[old], fixed[new] - fixed[old], refitted[new] - refitted[old])))
    recovery = {key: {"mean": value, "fixed_pipeline_95_interval": np.quantile(construct_fixed[key], [.025, .975]).tolist(),
                      "joint_refit_95_interval": np.quantile(construct_refit[key], [.025, .975]).tolist()}
                for key, value in construct_point.items()}
    recovery_comparisons = [dict(name=key, **summarize_difference(construct_point[key], construct_fixed[key], construct_refit[key]))
                            for key in RECOVERY_COMPARISONS]
    np.savez_compressed(args.output / f"private_{language}_draws.npz",
        **{"fixed__" + k: v for k, v in fixed.items()}, **{"refit__" + k: v for k, v in refitted.items()},
        **{"construct_fixed__" + k: np.asarray(v) for k, v in construct_fixed.items()},
        **{"construct_refit__" + k: np.asarray(v) for k, v in construct_refit.items()},
        **{"coefficient__" + k: np.asarray(v) for k, v in coefficient_draws.items()},
        temperatures=np.asarray(temperature_draws), full_temperatures=np.asarray(full_temperatures))
    (args.output / f"private_{language}_draw_provenance.json").write_text(json.dumps({
        "seed": 20261009, "sorted_documents": groups.tolist(), "multiplicity_sha256": draw_ids,
        "opportunity_item_ids": frame.item_id.tolist(), "form_item_ids": frame.loc[selected, "item_id"].tolist(),
        "folds": folds.tolist()}, indent=2) + "\n")
    rows = frame.loc[selected, ["item_id", "doc_id", "fold", "gold_form", "y"]].copy()
    for name, prediction in point_predictions.items():
        rows[name] = prediction
    rows.to_csv(args.output / f"private_{language}_predictions.csv", index=False)
    metrics = {name: {"nll": point[name], "brier": float(np.average((p - y) ** 2, weights=form_weights)),
                      "accuracy": float(np.average((p >= .5) == y, weights=form_weights))} for name, p in point_predictions.items()}
    coefficient_summary = {name: {"feature_names": HISTORY + ["P_surprisal", "P_entropy", "log_hidden_mentions", "DP_surprisal_increment", "DP_entropy_increment"],
        "intercept_and_coefficients": point_coef[name].tolist(),
        "joint_refit_95_intervals": np.quantile(coefficient_draws[name], [.025, .975], axis=0).T.tolist()}
        for name in point_coef}
    return {"language": language, "opportunities": len(frame), "calibration_documents": len(groups),
        "form_rows": len(selected), "form_documents": int(frame.loc[selected, "doc_id"].nunique()),
        "fixed_draws": 2000, "refit_draws": 100, "metrics": metrics, "form_comparisons": form_comparisons,
        "recovery": recovery, "recovery_comparisons": recovery_comparisons, "coefficients": coefficient_summary,
        "temperature_fits": original_temperatures, "full_descriptive_inverse_temperature": full_alpha,
        "joint_refit_temperature_summary": {"scalars": int(np.asarray(temperature_draws).size),
            "minimum": float(np.min(temperature_draws)), "maximum": float(np.max(temperature_draws)),
            "uniform_boundary_count": int(np.equal(temperature_draws, 0.).sum()),
            "upper_boundary_count": int(np.equal(temperature_draws, 5.).sum())},
        "raw_archive_reproduction": reproduction, "fits": fits, "full_fits": full_fits,
        "numerics": {**numerical, "solver_fallbacks": fallbacks}, "old_bootstrap_draws_reproduced": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--reference-analysis", type=Path, required=True)
    parser.add_argument("--local-history", type=Path, required=True)
    parser.add_argument("--boundary", type=Path, required=True)
    parser.add_argument("--model-name", choices=tuple(MODELS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    protocol = HERE / "README.md"
    protocol_hash = file_hash(protocol)
    data = load_data(args)
    args.output.mkdir(parents=True)
    manifest = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "model": args.model_name,
        "source_scores": str(args.scores), "source_scores_sha256": file_hash(args.scores),
        "readme_sha256": protocol_hash, "implementation_sha256": {str(p): file_hash(p) for p in
            (Path(__file__).resolve(), Path(entity_analysis.__file__).resolve())},
        "input_sha256": file_hash(HERE / "work/probes/inputs.jsonl"),
        "items_sha256": file_hash(HERE / "work/probes/items.jsonl"),
        "reference_features_sha256": file_hash(args.reference_analysis / "features.csv"),
        "local_history_summary_sha256": file_hash(args.local_history / "summary.json"),
        "boundary_records_sha256": file_hash(args.boundary / "private_boundary_records.csv"),
        "score_job_source_hashes_as_recorded": data["scoring_source_hashes"], "interrupted_prefix_rows": data["prefix_rows"],
        "raw_measurement_reproduction": data["raw_measurement_reproduction"],
        "post_result": True, "new_calibrated_outcomes_computed": False, "seed": 20261009}
    (args.output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    results = {language: run_language(data, language, args) for language in ("ca", "cs", "es")}
    assert file_hash(protocol) == protocol_hash
    report = {"complete": True, "model": args.model_name, "readme_sha256": protocol_hash, "results": results,
        "post_result_sensitivity": True, "original_primary_replaced": False,
        "pending_families": "separate 12-slot form and 12-slot referent-recovery families after both checkpoints",
        "uncertainty": "fixed document folds and LLM scores; full calibration-document draws jointly refit temperature and form heads"}
    (args.output / "summary.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "model": args.model_name, "languages": 3, "joint_refits_per_language": 100}), flush=True)


if __name__ == "__main__":
    main()
