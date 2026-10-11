"""Evaluate entity recovery and subject-form prediction on document folds."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
from entity_analysis import read_jsonl,read_probes,gold_features,grouped_evaluation,contrast,holm,cluster_summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--refits",type=int,default=100)
    args = parser.parse_args()
    assert not args.output.exists()
    probes = read_probes(HERE/"work/probes",args.scores)
    rows,recovery = [],[]
    for record in read_jsonl(HERE/"work/probes/items.jsonl"):
        row = {key:value for key,value in record.items() if key not in ("prefix","candidates")}
        row["y"] = int(record["gold_form"]=="np")
        row["log_candidates"] = float(np.log1p(record["candidate_count"]))
        aliases = [c["alias"]["text"].strip().casefold() for c in record["candidates"]]
        row["alias_collision"] = int(len(set(aliases))!=len(aliases))
        for name in ("matching_count","unknown_count","ambiguous_count"):
            row["log_"+name] = float(np.log1p(record[name]))
        for condition in ("D","DP","P"):
            values = gold_features(probes[(record["item_id"],"subject",condition)],record["gold_entity"])
            row.update({condition+"_"+key:value for key,value in values.items()})
            probe = probes[(record["item_id"],"subject",condition)]
            for status in ("compatible","unknown","ambiguous"):
                row[condition+"_competitor_mass_"+status] = sum(probe["probabilities"][c["entity_id"]] for c in record["candidates"]
                    if c["entity_id"]!=record["gold_entity"] and c["number_match"]==status)
            recovery.append({"item_id":record["item_id"],"language":record["language"],"doc_id":record["doc_id"],
                             "gold_form":record["gold_form"],"condition":condition,"candidates":record["candidate_count"],
                             "uniform_nll":float(np.log(record["candidate_count"])),
                             "uniform_accuracy":1/record["candidate_count"],
                             "recency_correct":int(record["candidates"][0]["entity_id"]==record["gold_entity"]),**values})
        for condition in ("D","DP"):
            row[condition+"_surprisal_increment"] = row[condition+"_surprisal"]-row["P_surprisal"]
            row[condition+"_entropy_increment"] = row[condition+"_entropy"]-row["P_entropy"]
        for status in ("compatible","unknown","ambiguous"):
            row["competitor_mass_increment_"+status] = row["DP_competitor_mass_"+status]-row["P_competitor_mass_"+status]
        rows.append(row)
    frame = pd.DataFrame(rows)
    baseline = ["log_distance","log_prior_mentions","prior_subject","log_candidates","log_matching_count",
                "log_unknown_count","log_ambiguous_count","current_plural","aux_no_part","has_part","clause_coord","clause_embedded"]
    prior = baseline+["P_surprisal","P_entropy"]
    columns = {"H":baseline,"HP":prior,"HPC":prior+["DP_surprisal_increment","DP_entropy_increment"],
               "HPD":prior+["D_surprisal_increment","D_entropy_increment"]}
    columns["HPM"] = prior+["P_competitor_mass_"+s for s in ("compatible","unknown","ambiguous")]
    columns["HPMC"] = columns["HPM"]+["competitor_mass_increment_"+s for s in ("compatible","unknown","ambiguous")]
    args.output.mkdir(parents=True)
    frame.to_csv(args.output/"features.csv",index=False)
    recovery = pd.DataFrame(recovery)
    recovery["nll_minus_uniform"] = recovery.surprisal-recovery.uniform_nll
    recovery["accuracy_minus_uniform"] = recovery.correct-recovery.uniform_accuracy
    recovery["accuracy_minus_recency"] = recovery.correct-recovery.recency_correct
    recovery.to_csv(args.output/"recovery.csv",index=False)
    constructs = {}
    recovery_columns = ["surprisal","correct","nll_minus_uniform","accuracy_minus_uniform","accuracy_minus_recency",
                        "other_mass","answer_mass","order_total_variation","order_top_disagreement"]
    for (language,condition),part in recovery.groupby(["language","condition"]):
        constructs[language+":"+condition] = cluster_summary(part,recovery_columns,"doc_id",True)
    context_construct = {}
    for language,part in frame.groupby("language"):
        part = part.copy()
        part["recovery_accuracy_increment"] = part.DP_correct-part.P_correct
        part["one_entity_set"] = part.candidate_count.eq(1).astype(int)
        context_construct[language] = cluster_summary(part,["DP_surprisal_increment","recovery_accuracy_increment",
                                                          "one_entity_set","alias_collision"],"doc_id",True)
    summaries,primary,predictions = {},[],[]
    for language,group in frame[frame.gold_form.isin(["null","np"])].groupby("language"):
        group = group.reset_index(drop=True)
        result = grouped_evaluation(group,columns,"doc_id","fold",True,args.refits,20261009)
        summary = {key:result[key] for key in ("groups","rows","weighting","metrics","fits","full_fits","numerics")}
        summary["comparisons"] = [contrast(result,*pair) for pair in [("HPC","HP"),("HP","H"),("HPC","HPD"),("HPMC","HPM")]]
        summaries[language] = summary
        primary.append(summary["comparisons"][0])
        prediction = group[["item_id","doc_id","language","fold","gold_form","y"]].copy()
        for name,p in result["predictions"].items():
            prediction[name] = p
        predictions.append(prediction)
        np.savez_compressed(args.output/f"{language}_bootstrap.npz",**{"fixed_"+k:v for k,v in result["fixed_draws"].items()},
                            **{"refit_"+k:v for k,v in result["refit_draws"].items()})
        print(json.dumps({"language":language,"main_comparison":primary[-1]}),flush=True)
        sensitivity = {}
        for name,subset in [("multi_candidate_zero_np",group[group.candidate_count.gt(1)]),
                            ("zero_vs_all_overt",frame[frame.language.eq(language)].assign(y=lambda d:d.gold_form.ne("null").astype(int)))]:
            alternative = grouped_evaluation(subset.reset_index(drop=True),{k:columns[k] for k in ("HP","HPC")},
                                             "doc_id","fold",True,0,20261009)
            sensitivity[name] = {"rows":alternative["rows"],"groups":alternative["groups"],
                                 "comparison":contrast(alternative,"HPC","HP"),"metrics":alternative["metrics"],
                                 "fits":alternative["fits"],"full_fits":alternative["full_fits"],"numerics":alternative["numerics"]}
        summary["secondary_sensitivities"] = sensitivity
    for row,adjusted in zip(primary,holm([r["normal_approximation_p"] for r in primary])):
        row["holm_p_three_languages"] = adjusted
    pd.concat(predictions,ignore_index=True).to_csv(args.output/"heldout_predictions.csv",index=False)
    (args.output/"summary.json").write_text(json.dumps({"source_scores":str(args.scores),"results":summaries,
        "interpretation":"conditional corpus associations; clustered intervals under document reweighting", "C":1,
        "construct_checks":constructs,"context_construct_checks":context_construct,
        "primary":"HPC-HP in each language; Qwen primary, Euro prespecified replication", "completed":True},indent=2)+"\n")


if __name__ == "__main__":
    main()
