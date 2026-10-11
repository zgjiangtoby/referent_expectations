"""Build entity probes with the observed subject form hidden."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CODE = Path(__file__).resolve().parent
PROJECT = CODE
from extract_candidates import read_corpus, subject_links
from measure_features import projection_sources, project, match


LANGUAGES = ("ca", "es", "cs")
EXPECTED = {"ca": (12580,2906), "es": (14513,2986), "cs": (12139,3134)}


def read_jsonl(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def resolve(path):
    path = Path(path)
    return path if path.is_absolute() else PROJECT/path


def source_texts(path):
    sentences, doc, sent, text, rows = {}, None, None, None, []
    def store():
        if sent is None:
            return
        surface, covered = [], set()
        for fields in rows:
            token_id = fields[0]
            if "-" in token_id:
                lo, hi = map(int,token_id.split("-"))
                covered.update(range(lo,hi+1))
                surface.append(fields[1]+("" if "SpaceAfter=No" in fields[9] else " "))
            elif "." not in token_id and int(token_id) not in covered:
                surface.append(fields[1]+("" if "SpaceAfter=No" in fields[9] else " "))
        sentences[(doc,sent)] = {"text":text if text is not None else "".join(surface).strip(),
                                 "index":len(sentences),"doc":doc}
    with path.open() as stream:
        for line in stream:
            if line.startswith("# newdoc id = "):
                doc = line.strip().split(" = ",1)[1]
            elif line.startswith("# sent_id = "):
                sent = line.strip().split(" = ",1)[1]
            elif line.startswith("# text = "):
                text = line.rstrip("\n").split(" = ",1)[1]
            elif line.strip() and not line.startswith("#"):
                rows.append(line.rstrip("\n").split("\t"))
            elif not line.strip() and rows:
                store()
                sent, text, rows = None,None,[]
    if rows:
        store()
    return sentences


def alias(mentions):
    overt = [m for m in mentions if any(not w["empty"] and w["form"] not in ("_","") for w in m["words"])]
    lexical = [m for m in overt if m["head"]["upos"] in ("NOUN","PROPN") and not m["head"]["empty"]]
    if not overt:
        return None
    selected = max(lexical or overt,key=lambda m:(m["end"],m["start"]))
    words = [w["form"] for w in selected["words"] if not w["empty"] and w["form"] not in ("_","")]
    return {"text":" ".join(words[:24])+(" ..." if len(words)>24 else ""),
            "alias_lexical":bool(lexical),"alias_sentence":selected["head"]["sentence"]["id"],
            "alias_head":selected["head"]["id"],"alias_start":selected["start"],"alias_end":selected["end"]}


def make_prompt(prefix,candidates,predicate,number,condition,order):
    arranged = list(range(len(candidates)))
    if order:
        arranged.reverse()
    labels = list("ABCDEFGHIJKLM")[:len(candidates)+1]
    options = []
    for letter,index in zip(labels,arranged):
        candidate = candidates[index]
        options.append(f'{letter}. "{candidate["alias"]["text"]}" (last mention: sentence {candidate["sentence_offset"]}, token {candidate["head_token"]})')
    options.append(f"{labels[-1]}. Another entity not listed")
    prompt = ("Select the entity most likely to be the grammatical subject of a forthcoming clause. "
              "Each listed option denotes a distinct discourse entity. The wording is a previously observed mention, "
              "not the forthcoming referring expression. Sentence offsets are relative to the forthcoming sentence.\n")
    if condition in ("D","DP"):
        prompt += "\nPRECEDING DISCOURSE:\n"+prefix+"\n"
    if condition in ("P","DP"):
        prompt += f"\nForthcoming predicate lemma: {predicate}\nFinite grammatical number: {number}\n"
    prompt += "\nENTITIES:\n"+"\n".join(options)+"\n\nReturn only the letter."
    return prompt,labels,[candidates[i]["entity_id"] for i in arranged]+["OTHER"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurement-root",type=Path,default=CODE/"work/preparation/measurement")
    parser.add_argument("--analysis-root",type=Path,default=CODE/"work/preparation/analysis")
    parser.add_argument("--output",type=Path,required=True)
    args = parser.parse_args()
    assert not args.output.exists(),args.output
    args.output.mkdir(parents=True)
    rng = np.random.default_rng(20261009)
    report, records = {},[]
    for language in LANGUAGES:
        root = args.measurement_root/language
        source_rows = read_jsonl(args.analysis_root/f"{language}_rows.jsonl")
        assert (len(source_rows),sum(row["eligible"] for row in source_rows)) == EXPECTED[language]
        by_split = defaultdict(list)
        for row in source_rows:
            by_split[row["split"]].append(row)
        docs = sorted({row["doc_id"] for row in source_rows})
        folds = {docs[index]:i%5 for i,index in enumerate(rng.permutation(len(docs)))}
        exclusions, included, recovered = Counter(),Counter(),0
        for split,group in sorted(by_split.items()):
            coverage = json.loads((root/split/"coverage.json").read_text())
            raw = resolve(coverage["source_file"])
            reference = {(row["doc_id"],row["sent_id"],row["token_id"]):row
                         for row in read_jsonl(resolve(coverage["canonical_v3"])/"candidates.jsonl")}
            mentions,_ = read_corpus(raw)
            by_doc = defaultdict(list)
            for mention in mentions:
                by_doc[mention["doc"]].append(mention)
            texts = source_texts(raw)
            indexed_texts = {value["index"]:value for value in texts.values()}
            description_cache = {}
            for row in group:
                recovered += 1
                ref = reference[(row["doc_id"],row["sent_id"],row["token_id"])]
                target_mentions = [m for m in by_doc[row["doc_id"]]
                                   if m["head"]["sentence"]["id"] == row["sent_id"] and m["head"]["id"] == row["token_id"]
                                   and m["meta"]["eid"] == row["entity_id"]]
                assert len(target_mentions) == 1,(language,split,row["sent_id"],row["token_id"])
                sentence = target_mentions[0]["head"]["sentence"]
                current = sentence["index"]
                assert texts[(row["doc_id"],row["sent_id"])]["index"] == current
                cutoff = min(node["position"] for node in sentence["nodes"].values())
                prior = [m for m in by_doc[row["doc_id"]] if m["end"] < cutoff]
                recent = [m for m in prior if min(w["sentence"]["index"] for w in m["words"]) >= current-2]
                history = defaultdict(list)
                for mention in recent:
                    history[mention["meta"]["eid"]].append(mention)
                candidates = []
                for entity,chain in history.items():
                    latest = max(chain,key=lambda m:(m["end"],m["start"]))
                    name = alias(chain)
                    if name is None:
                        exclusions["candidate_without_surface_alias"] += 1
                        continue
                    if id(latest) not in description_cache:
                        description_cache[id(latest)] = projection_sources(latest,language)
                    projection = project(latest,description_cache[id(latest)],cutoff)
                    state = match(projection["alternatives"],[{"Number":"Plur" if row["current_plural"] else "Sing"}])
                    candidates.append({"entity_id":entity,"alias":name,"end":latest["end"],
                                       "head_token":latest["head"]["id"],
                                       "sentence_offset":latest["head"]["sentence"]["index"]-current,
                                       "number_match":state})
                candidates.sort(key=lambda c:(-c["end"],c["entity_id"]))
                candidates = candidates[:12]
                if row["entity_id"] not in {c["entity_id"] for c in candidates}:
                    exclusions["gold_not_in_pool:"+row["form"]] += 1
                    continue
                gold_prior = [m for m in prior if m["meta"]["eid"] == row["entity_id"]]
                assert gold_prior
                latest_gold = max(gold_prior,key=lambda m:(m["end"],m["start"]))
                matching = Counter(c["number_match"] for c in candidates if c["entity_id"] != row["entity_id"])
                prefix = "\n".join(indexed_texts[i]["text"] for i in (current-2,current-1)
                                   if i in indexed_texts and indexed_texts[i]["doc"] == row["doc_id"])
                assert prefix and all(c["alias"]["alias_end"] < cutoff for c in candidates)
                item_id = f"referent:{language}:{split}:{row['doc_id']}:{row['sent_id']}:{row['token_id']}"
                record = {"item_id":item_id,"language":language,"split":split,"doc_id":row["doc_id"],
                          "sent_id":row["sent_id"],"token_id":row["token_id"],"gold_entity":row["entity_id"],
                          "gold_form":row["form"],"fold":folds[row["doc_id"]],"prefix":prefix,
                          "candidates":candidates,"predicate_lemma":ref["predicate_lemma"],
                          "finite_number":"Plur" if row["current_plural"] else "Sing",
                          "sentence_cutoff_position":cutoff,"log_distance":float(np.log1p(cutoff-latest_gold["end"])),
                          "log_prior_mentions":float(np.log1p(len(gold_prior))),
                          "prior_subject":int(bool(subject_links(latest_gold["head"]))),
                          "candidate_count":len(candidates),"matching_count":matching["compatible"],
                          "unknown_count":matching["unknown"],"ambiguous_count":matching["ambiguous"],
                          **{name:row[name] for name in ("current_plural","aux_no_part","has_part","clause_coord","clause_embedded")}}
                records.append(record)
                included[row["form"]] += 1
            print(json.dumps({"language":language,"source_split":split,"source_rows":len(group),"included_so_far":sum(included.values())}),flush=True)
        assert recovered == len(source_rows)
        report[language] = {"current_source_rows":recovered,"current_complete_rows":EXPECTED[language][1],
                            "included_forms":dict(included),"exclusions":dict(exclusions),
                            "measurement_root":str(root),"source_folds":dict(Counter(r["fold"] for r in records if r["language"]==language))}
    technical = set()
    for language in LANGUAGES:
        pool = [r["item_id"] for r in records if r["language"]==language and r["split"]=="dev"]
        assert len(pool)>=12
        technical.update(rng.choice(pool,size=12,replace=False).tolist())
    with (args.output/"items.jsonl").open("x") as items,(args.output/"inputs.jsonl").open("x") as full,(args.output/"pilot.jsonl").open("x") as pilot:
        for record in records:
            record["technical_pilot"] = record["item_id"] in technical
            items.write(json.dumps(record,ensure_ascii=False)+"\n")
            for condition in ("D","DP","P"):
                for order in (0,1):
                    prompt,labels,entities = make_prompt(record["prefix"],record["candidates"],record["predicate_lemma"],record["finite_number"],condition,order)
                    row = {"id":f"{record['item_id']}:{condition}:o{order}","prompt":prompt,"labels":labels,
                           "item_id":record["item_id"],"condition":condition,"order":order,"entity_codes":entities}
                    line = json.dumps(row,ensure_ascii=False)+"\n"
                    full.write(line)
                    if record["technical_pilot"]:
                        pilot.write(line)
    report["total_included"] = len(records)
    report["technical_pilot"] = len(technical)
    (args.output/"summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report),flush=True)


if __name__ == "__main__":
    main()
