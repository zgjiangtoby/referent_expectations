#!/usr/bin/env python3
"""Project annotated antecedent features and measure local competition."""
import argparse
import json
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

from extract_candidates import (feature_alternatives, finite_construction,
                                read_corpus, subject_links)

FEATURES = ('Person', 'Number', 'Gender')
PROTOCOL = 'measurement_v4_1'


def read_jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def mention_key(mention):
    head = mention['head']
    return (mention['doc'], head['sentence']['id'], head['id'], mention['meta']['eid'],
            tuple(f"{word['sentence']['id']}/{word['id']}" for word in mention['words']))


def row_key(row):
    return (row['doc_id'], row['sent_id'], row['token_id'], row['entity_id'],
            tuple(row['mention_node_ids']))


def source(node, rule, keys=FEATURES):
    alternatives = [{k: v for k, v in values.items() if k in keys}
                    for values in feature_alternatives(node)]
    return {'node': f"{node['sentence']['id']}/{node['id']}",
            'position': node['position'], 'rule': rule,
            'raw_feats': node['feats'], 'raw_xpos': node['xpos'],
            'alternatives': alternatives}


def projection_sources(mention, language):
    """Own-mention evidence only; target-specific temporal cutoff comes later."""
    head = mention['head']
    nodes = head['sentence']['nodes']
    reasons = []
    nominal = head['upos'] in ('NOUN', 'PROPN', 'PRON')
    lexical = head['upos'] in ('NOUN', 'PROPN') and not head['empty']
    coordinated = any(node['head'] == head['id'] and node['rel'].split(':')[0] == 'conj'
                      and node['sentence'] is head['sentence'] for node in mention['words'])
    if coordinated:
        return [], ['mention_internal_coordination'], False, False
    if head['feats'].get('Poss') == 'Yes':
        if len(mention['words']) == 1 and head['upos'] in ('DET', 'ADJ', 'PRON') and (
                head['rel'].split(':')[0] in ('det', 'amod') or head['rel'] == 'nmod:poss'):
            mapped = {key: head['feats'][raw] for key, raw in (
                ('Number', 'Number[psor]'), ('Gender', 'Gender[psor]'), ('Person', 'Person'))
                if raw in head['feats']}
            keys = list(mapped)
            evidence = source(head, 'published_possessor_features')
            evidence['alternatives'] = [dict(zip(keys, values)) for values in
                                        product(*(mapped[key].split(',') for key in keys))]
            return [evidence], ['possessive_modifier_uses_psor_features_only'], False, False
        return [], ['possessive_referent_layer_unresolved'], False, False
    if not nominal and not head['empty']:
        return [], ['head_not_nominal_or_empty_subject'], False, False
    sources = [source(head, 'raw_mention_head')]
    links = subject_links(head)
    independent = len(links) == 1 and 'xsubj' not in links[0][1]
    if not independent:
        reasons.append('no_unique_noncontrolled_subject_governor')
    else:
        pid, relation = links[0]
        predicate = nodes.get(pid)
        others = [node for node in nodes.values() if node is not head
                  and any(gov == pid and 'xsubj' not in rel for gov, rel in subject_links(node))]
        conjoined_head = head['rel'].split(':')[0] == 'conj' or any(
            node['head'] == head['id'] and node['rel'].split(':')[0] == 'conj'
            for node in nodes.values())
        if others or conjoined_head:
            reasons.append('shared_or_coordinated_subject_no_projection')
        elif predicate is None or predicate['upos'] not in ('VERB', 'AUX'):
            reasons.append('nonverbal_predicate_no_projection')
        else:
            construction, why = finite_construction(predicate, nodes, language)
            if construction is None:
                reasons.append('own_construction_' + why)
            else:
                for node in construction['cluster']:
                    feats = node['feats']
                    if node is construction['anchor'] and feats.get('VerbForm') == 'Fin':
                        sources.append(source(node, 'own_finite_subject_agreement', ('Person', 'Number')))
                    elif language == 'cs' and node['upos'] in ('VERB', 'AUX') and (
                            node['xpos'][:2] in ('Vp', 'Vq') and feats.get('VerbForm') == 'Part'
                            and feats.get('Tense') == 'Past' and feats.get('Voice') == 'Act'):
                        sources.append(source(node, 'own_czech_lpart_agreement', ('Number', 'Gender')))
                    elif node is predicate and feats.get('VerbForm') == 'Part':
                        passive_aux = any(
                            (aux['head'] == pid and aux['rel'] == 'aux:pass') or
                            any(gov == pid and rel == 'aux:pass' for gov, rel in aux['deps'])
                            for aux in construction['cluster'] if aux is not predicate)
                        if ':pass' in relation and passive_aux:
                            sources.append(source(node, 'own_passive_part_agreement', ('Number', 'Gender')))
    span_non_third = any(node['upos'] == 'PRON' and
                         set(node['feats'].get('Person', '').split(',')) & {'1', '2'}
                         for node in mention['words'])
    return sources, reasons, lexical, bool(span_non_third)


def merge_sources(sources):
    values = {}
    conflicts = []
    for feature in FEATURES:
        observations = [{item[feature] for item in evidence['alternatives'] if feature in item}
                        for evidence in sources]
        observations = [item for item in observations if item]
        if not observations:
            continue
        intersection = set.intersection(*observations)
        if intersection:
            values[feature] = sorted(intersection)
        else:
            conflicts.append(feature)
    keys = list(values)
    alternatives = [dict(zip(keys, row)) for row in product(*(values[key] for key in keys))]
    # Preserve dependencies internal to each source, including PDT Q/W.
    alternatives = [row for row in alternatives if all(any(
        all(key not in row or row[key] == val for key, val in alt.items())
        for alt in evidence['alternatives']) for evidence in sources)]
    if not alternatives:
        return [{}], sorted(set(conflicts) | set(keys)), True
    return alternatives, conflicts, False


def project(mention, description, cutoff):
    sources, reasons, lexical, span_non_third = description
    reasons = list(reasons)
    usable = [evidence for evidence in sources if evidence['position'] < cutoff]
    delayed = [evidence['node'] for evidence in sources if evidence['position'] >= cutoff]
    if delayed:
        reasons.append('own_agreement_source_after_target_cutoff')
    head_person = mention['head']['feats'].get('Person')
    agreement_non_third = any(set(alt.get('Person', '').split(',')) & {'1', '2'}
                              for evidence in usable if evidence['rule'] != 'raw_mention_head'
                              for alt in evidence['alternatives'])
    if lexical and head_person is None:
        if span_non_third or agreement_non_third:
            reasons.append('lexical_np_default_third_blocked_non_third_evidence')
            usable = [{**evidence, 'alternatives': [{k: v for k, v in alt.items() if k != 'Person'}
                       for alt in evidence['alternatives']]} for evidence in usable]
        else:
            usable.append({'node': f"{mention['head']['sentence']['id']}/{mention['head']['id']}",
                           'position': mention['head']['position'],
                           'rule': 'lexical_np_default_third', 'alternatives': [{'Person': '3'}]})
    alternatives, conflicts, joint_conflict = merge_sources(usable)
    reasons += ['conflicting_' + key for key in conflicts]
    if joint_conflict:
        reasons.append('joint_feature_conflict')
    states = {}
    for key in FEATURES:
        possibilities = sorted({alt[key] for alt in alternatives if key in alt})
        states[key] = {'values': possibilities, 'status': 'conflict' if key in conflicts else
                       'unknown' if not possibilities else 'known' if len(possibilities) == 1 else 'ambiguous'}
    return {'alternatives': alternatives, 'features': states, 'sources': usable,
            'delayed_source_nodes': delayed, 'reasons': reasons}


def match(alternatives, target_alternatives):
    """Certain match, certain contradiction, possible match, or missing evidence."""
    outcomes = set()
    for target in target_alternatives:
        for candidate in alternatives:
            if any(key in candidate and candidate[key] != val for key, val in target.items()):
                outcomes.add('incompatible')
            elif target and all(key in candidate for key in target):
                outcomes.add('compatible')
            else:
                outcomes.add('unknown')
    if outcomes == {'compatible'}:
        return 'compatible'
    if outcomes == {'incompatible'}:
        return 'incompatible'
    if 'unknown' in outcomes:
        return 'unknown'
    return 'ambiguous'


def count_matches(competitors, target):
    counter = Counter(match(alternatives, target) for alternatives in competitors)
    result = {key: counter[key] for key in ('compatible', 'incompatible', 'ambiguous', 'unknown')}
    result['lower'] = result['compatible']
    result['upper'] = result['compatible'] + result['ambiguous'] + result['unknown']
    result['fully_measured'] = result['ambiguous'] + result['unknown'] == 0
    return result


def run(canonical, output):
    inventory = json.loads((canonical / 'inventory.json').read_text())
    language = inventory['language']
    raw_path = Path(inventory['source_file'])
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(canonical / 'candidates.jsonl')
    unresolved = read_jsonl(canonical / 'unresolved_person_candidates.jsonl')
    # No Entity: canonical strict parser already established missing annotation.
    # The new feature pass cannot create a coreference layer from syntax.
    if not inventory['counts']['entity_token_rows']:
        summary = {'protocol': PROTOCOL, 'language': language, 'source_file': str(raw_path),
                   'coreference_annotation_status': 'absent_not_estimable', 'rows': 0,
                   'coverage': [], 'antecedent_feature_states': {}, 'projection_reasons': {}}
        (output / 'coverage.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary
    mentions, _ = read_corpus(raw_path)
    lookup = {mention_key(m): m for m in mentions}
    histories, by_doc = defaultdict(list), defaultdict(list)
    for idx, mention in enumerate(mentions):
        mention['projection_index'] = idx
        histories[(mention['doc'], mention['meta']['eid'])].append(mention)
        by_doc[mention['doc']].append(mention)
    descriptions, projections = {}, {}
    coverage = defaultdict(Counter)
    state_counts, reason_counts, source_counts = Counter(), Counter(), Counter()
    scope_counts = Counter()
    def get_projection(mention, cutoff):
        idx = mention['projection_index']
        if idx not in descriptions:
            descriptions[idx] = projection_sources(mention, language)
        description = descriptions[idx]
        available = tuple(item['position'] < cutoff for item in description[0])
        key = (idx, available)
        if key not in projections:
            result = project(mention, description, cutoff)
            result.update(projection_id=len(projections), doc_id=mention['doc'],
                          sent_id=mention['head']['sentence']['id'], token_id=mention['head']['id'],
                          entity_id=mention['meta']['eid'], head_upos=mention['head']['upos'],
                          head_empty=mention['head']['empty'], raw_head_feats=mention['head']['feats'],
                          mention_start=mention['start'], mention_end=mention['end'],
                          mention_discontinuous=mention['discontinuous'])
            projections[key] = result
        return projections[key]
    with (output / 'candidates.jsonl').open('w', encoding='utf-8') as out:
        for row in rows + unresolved:
            mention = lookup[row_key(row)]
            number = row['construction_anchor_feats'].get('Number')
            if language == 'ca' and row['form'] == 'pronoun' and row['subject_head_feats'].get('Poss') == 'Yes':
                scope = 'possessive_subject_outside_personal_pronoun'
            elif row['person_evidence'] == 'finite_carrier_Person_3':
                scope = 'primary' if number in ('Sing', 'Plur') else 'independent_person_number_unresolved'
            elif row['person_status'] == 'unresolved_person':
                scope = 'unresolved_person_extension'
            else:
                scope = 'subject_based_person_extension'
            scope_counts[scope] += 1
            # Preserve extensions separately, including Part forms with unknown person.
            if scope != 'primary':
                out.write(json.dumps({k: row[k] for k in (
                    'doc_id', 'sent_id', 'token_id', 'entity_id', 'form', 'person_evidence',
                    'construction_type', 'construction_has_participle', 'construction_has_aux_or_cop')}
                    | {'scope': scope}, ensure_ascii=False) + '\n')
                coverage[(scope, row['form'], 'all')]['rows'] += 1
                continue
            prior = [m for m in histories[(mention['doc'], mention['meta']['eid'])]
                     if m['end'] < mention['start']]
            last = max(prior, key=lambda m: m['end'])
            recent = {}
            for candidate in by_doc[mention['doc']]:
                if candidate['end'] >= mention['start'] or candidate['meta']['eid'] == mention['meta']['eid']:
                    continue
                if mention['head']['sentence']['index'] - candidate['head']['sentence']['index'] > 2:
                    continue
                eid = candidate['meta']['eid']
                if eid not in recent or candidate['end'] > recent[eid]['end']:
                    recent[eid] = candidate
            projected = [get_projection(m, mention['start']) for m in recent.values()]
            target_prior = get_projection(last, mention['start'])
            targets = [{'Number': number}]
            raw_number = count_matches([feature_alternatives(m['head']) for m in recent.values()], targets)
            projected_number = count_matches([p['alternatives'] for p in projected], targets)
            projected_full = count_matches([p['alternatives'] for p in projected], row['annotation_cue_alternatives'])
            raw_full = {'compatible': row['compatible_entities'], 'incompatible': row['incompatible_entities'],
                        'unknown': row['unknown_compatibility_entities']}
            profile = 'Fin_with_part' if row['construction_has_participle'] else (
                'Fin_aux_cop_no_part' if row['construction_has_aux_or_cop'] else 'Fin_single_no_part')
            result = {key: row[key] for key in (
                'doc_id', 'sent_id', 'token_id', 'entity_id', 'form', 'person_evidence',
                'construction_type', 'construction_has_participle', 'construction_has_aux_or_cop',
                'subject_relation', 'prior_mentions', 'prior_distance_positions', 'prior_distance_sentences',
                'prior_target_subject', 'predicate_lemma')}
            result.update(scope=scope, finite_number=number, recent_other_entities=len(recent),
                          prior_target_projection_id=target_prior['projection_id'],
                          competitor_projection_ids=[p['projection_id'] for p in projected],
                          raw_full_v3=raw_full, raw_number=raw_number,
                          projected_number=projected_number, projected_full=projected_full,
                          primary_model_eligible=projected_number['fully_measured'])
            out.write(json.dumps(result, ensure_ascii=False) + '\n')
            for group in ('all', profile):
                count = coverage[(scope, row['form'], group)]
                count['rows'] += 1
                count['no_other_entities'] += not recent
                count['competitor_occurrences'] += len(recent)
                count['raw_full_v3_any_unknown_rows'] += raw_full['unknown'] > 0
                for prefix, matches in (('raw_number', raw_number), ('projected_number', projected_number),
                                         ('projected_full', projected_full)):
                    count[prefix + '_fully_measured_rows'] += matches['fully_measured']
                    for status in ('compatible', 'incompatible', 'ambiguous', 'unknown'):
                        count[prefix + '_' + status + '_occurrences'] += matches[status]
                        count[prefix + '_any_' + status + '_rows'] += matches[status] > 0
                count['prior_target_number_known_rows'] += target_prior['features']['Number']['status'] == 'known'
            for p in projected:
                head_kind = 'empty' if p['head_empty'] else p['head_upos']
                for key, value in p['features'].items():
                    state_counts[(head_kind, key, value['status'])] += 1
                reason_counts.update(p['reasons'])
                source_counts.update(evidence['rule'] for evidence in p['sources'])
    with (output / 'antecedent_features.jsonl').open('w', encoding='utf-8') as out:
        for p in projections.values():
            out.write(json.dumps(p, ensure_ascii=False) + '\n')
    summary = {'protocol': PROTOCOL, 'language': language, 'source_file': str(raw_path),
               'canonical_v3': str(canonical), 'coreference_annotation_status': 'present',
               'rows': len(rows) + len(unresolved), 'scope_counts': dict(scope_counts),
               'distinct_projection_records': len(projections),
               'coverage': [dict(scope=scope, form=form, construction=kind, **count)
                            for (scope, form, kind), count in sorted(coverage.items())],
               'antecedent_feature_states': [dict(head_kind=kind, feature=key, status=status, count=n)
                    for (kind, key, status), n in sorted(state_counts.items())],
               'projection_reasons': dict(reason_counts), 'projection_sources': dict(source_counts),
               'occurrence_unit': 'one recent other entity per target opportunity; repeated histories counted',
               'model_fitted': False}
    (output / 'coverage.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--canonical', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.canonical, args.output)
    print(json.dumps({key: summary.get(key) for key in ('source_file', 'rows', 'scope_counts',
          'coreference_annotation_status', 'distinct_projection_records')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
