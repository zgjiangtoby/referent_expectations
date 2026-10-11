#!/usr/bin/env python3
"""Extract finite subject candidates and annotated entity histories from CorefUD."""
import argparse
import json
import re
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path


def read_corpus(path):
    """Adapt official UDAPI objects; UDAPI alone reconstructs mention pieces/heads."""
    from udapi.core.document import Document

    document = Document(str(path))
    schema = (document.meta.get('global.Entity') or '').split('-')
    counts = Counter()
    node_map = {}
    doc_id = None
    position = 0
    for sentence_index, root in enumerate(document.trees):
        if root.newdoc:
            doc_id = str(root.newdoc)
            counts['documents'] += 1
        if doc_id is None or not root.sent_id:
            raise ValueError('Explicit newdoc and sent_id boundaries required')
        sentence = {'id': root.sent_id, 'index': sentence_index, 'doc': doc_id, 'nodes': {}}
        counts['sentences'] += 1
        counts['multiword_rows'] += len(root.multiword_tokens)
        for raw in root.descendants_and_empty:
            tid = str(raw.ord)
            node = {'id': tid, 'form': raw.form, 'lemma': raw.lemma, 'upos': raw.upos,
                    'xpos': raw.xpos or '_', 'feats': dict(raw.feats),
                    'head': str(raw.parent.ord) if raw.parent is not None else '_',
                    'rel': raw.deprel or '_',
                    'deps': [d.split(':', 1) for d in (raw.raw_deps or '_').split('|') if ':' in d],
                    'misc': dict(raw.misc), 'sentence': sentence,
                    'position': position, 'empty': raw.is_empty()}
            node_map[raw] = node
            sentence['nodes'][tid] = node
            position += 1
            counts['empty_nodes' if node['empty'] else 'surface_tokens'] += 1
            counts['basic_nsubj_rows'] += node['rel'].split(':')[0] == 'nsubj'
            counts['entity_token_rows'] += bool(raw.misc['Entity'])
            counts['argtem_token_rows'] += bool(raw.misc['ArgTem'])
            counts['raw_mention_parts'] += raw.misc['Entity'].count('(')
            counts['raw_discontinuous_parts'] += len(re.findall(r'\([^()]*?\[\d+/\d+\]', raw.misc['Entity']))
    counts['conllu_rows'] = counts['surface_tokens'] + counts['empty_nodes'] + counts['multiword_rows']
    # Some native PDT shards contain only m/a-layer annotation (e.g. train-ca).
    # Preserve their real corpus inventory, but do not invent an identity layer.
    if not counts['entity_token_rows']:
        counts.update(mentions=0, tracked_identity_mentions=0)
        return [], counts
    if not {'eid', 'head'} <= set(schema):
        raise ValueError('Expected the published eid/head CorefUD schema when Entity annotations are present')
    # The upstream loader uses strict=True: no permissive repair/deletion mode.
    mentions = document.coref_mentions
    counts['mentions'] = len(mentions)
    retained = []
    discontinuous_entities = set()
    for mention in mentions:
        eid = mention.entity.eid
        if eid.startswith('NOCOREF'):
            counts['excluded_nocoref_mentions'] += 1
            continue
        words = [node_map[word] for word in mention.words]
        head = node_map[mention.head]
        if len({word['sentence']['doc'] for word in words}) != 1:
            raise ValueError('An identity mention crosses document boundaries')
        positions = [word['position'] for word in words]
        discontinuous = len(positions) != max(positions) - min(positions) + 1
        item = {'meta': {'eid': eid, 'etype': mention.other.get('orig_etype', mention.entity.etype or '')},
                'words': words, 'doc': head['sentence']['doc'], 'head': head,
                'start': min(positions), 'end': max(positions), 'discontinuous': discontinuous}
        if discontinuous:
            counts['discontinuous_identity_mentions'] += 1
            discontinuous_entities.add((item['doc'], eid))
        retained.append(item)
    for mention in retained:
        mention['entity_has_discontinuous_mentions'] = (
            mention['doc'], mention['meta']['eid']) in discontinuous_entities
    counts['entities_with_discontinuous_mentions'] = len(discontinuous_entities)
    counts['tracked_identity_mentions'] = len(retained)
    return sorted(retained, key=lambda m: (m['start'], m['end'])), counts


def subject_links(node):
    # Enhanced links recover null subjects; propagated xsubj is not a free choice.
    links = [(node['head'], node['rel'])] if not node['empty'] else []
    links += [tuple(d) for d in node['deps']]
    by_head = {}
    for head, relation in links:
        if relation.split(':')[0] != 'nsubj':
            continue
        # Basic and enhanced descriptions of the same arc are one governor.
        # Prefer an explicit control subtype when present.
        if head not in by_head or 'xsubj' in relation:
            by_head[head] = relation
    direct = [(head, rel) for head, rel in by_head.items() if 'xsubj' not in rel]
    return sorted(direct or by_head.items())


def verbal_complex(predicate, nodes, language):
    """Use annotated aux/cop arcs, including Czech enhanced shared auxiliaries."""
    cluster = [predicate]
    for node in nodes.values():
        arcs = [(node['head'], node['rel'])]
        if language == 'cs':
            arcs += [tuple(arc) for arc in node['deps']]
        if any(head == predicate['id'] and rel.split(':')[0] in ('aux', 'cop')
               for head, rel in arcs):
            cluster.append(node)
    return cluster


def finite_construction(predicate, nodes, language):
    if predicate is None:
        return None, 'missing_predicate'
    cluster = verbal_complex(predicate, nodes, language)
    finite = [node for node in cluster if node['feats'].get('VerbForm') == 'Fin']
    if len(finite) == 1:
        anchor = finite[0]
        kind = 'overt_finite'
    elif len(finite) > 1:
        return None, 'multiple_finite_carriers'
    elif language != 'cs':
        return None, 'missing_finite_carrier'
    else:
        # The active past l-participle is not an attributive/present/passive Part.
        lparts = [node for node in cluster
                  if node['upos'] in ('VERB', 'AUX') and node['xpos'][:2] in ('Vp', 'Vq')
                  and node['feats'].get('VerbForm') == 'Part'
                  and node['feats'].get('Tense') == 'Past'
                  and node['feats'].get('Voice') == 'Act']
        if len(lparts) != 1:
            return None, 'missing_or_multiple_czech_l_participles'
        anchor = lparts[0]
        if predicate['rel'].split(':')[0] not in ('root', 'ccomp', 'csubj', 'advcl', 'acl', 'parataxis', 'conj'):
            return None, 'czech_part_outside_selected_clause_relations'
        if any(node['feats'].get('VerbForm') in ('Inf', 'Conv') for node in cluster):
            return None, 'czech_part_in_nonfinite_complex'
        if anchor is predicate:
            kind = 'czech_past_l_participle'
        elif anchor['lemma'] == 'být':
            kind = 'czech_past_byt_aux_or_cop'
        else:
            return None, 'unrecognized_czech_part_auxiliary'
    cue_nodes = [anchor]
    if language == 'cs':
        cue_nodes += [node for node in cluster if node is not anchor
                      and node['feats'].get('VerbForm') == 'Part']
    return {'kind': kind, 'anchor': anchor, 'cluster': cluster, 'cue_nodes': cue_nodes}, None


def third_person_evidence(construction, subject, language):
    """Resolve grammatical person without using antecedent identity or lexical form."""
    # A nominal predicate's Person describes that predicate (e.g. "of us"),
    # not agreement with its subject. Only verbal-form carriers contribute.
    persons = {node['feats']['Person'] for node in construction['cluster']
               if 'Person' in node['feats'] and node['feats'].get('VerbForm')}
    subject_person = subject['feats'].get('Person')
    if subject_person is not None:
        persons.add(subject_person)
    if persons - {'3'}:
        return 'not_confirmed_third', 'existing_person_annotation_not_3'
    anchor = construction['anchor']
    if anchor['feats'].get('VerbForm') == 'Fin' and anchor['feats'].get('Person') == '3':
        return 'confirmed_third', 'finite_carrier_Person_3'
    if language == 'cs' and subject_person == '3':
        return 'confirmed_third', 'subject_Person_3'
    if language == 'cs' and subject['upos'] in ('NOUN', 'PROPN'):
        return 'confirmed_third', 'lexical_nominal_subject_grammatical_person_rule'
    return 'unresolved_person', 'no_independent_person_annotation'


def feature_alternatives(node):
    """Retain the documented PDT Q/W gender-number dependency."""
    feats = {key: node['feats'][key] for key in ('Person', 'Number', 'Gender') if key in node['feats']}
    keys = list(feats)
    alternatives = [dict(zip(keys, values)) for values in product(*(feats[key].split(',') for key in keys))]
    if node['xpos'][2:4] == 'QW' and {'Gender', 'Number'} <= feats.keys():
        alternatives = [values for values in alternatives
                        if (values['Gender'], values['Number']) in (('Fem', 'Sing'), ('Neut', 'Plur'))]
    return alternatives


def construction_cues(construction):
    alternatives = [{}]
    for node in construction['cue_nodes']:
        combined = []
        for previous in alternatives:
            for current in feature_alternatives(node):
                if all(key not in previous or previous[key] == value for key, value in current.items()):
                    merged = {**previous, **current}
                    if merged not in combined:
                        combined.append(merged)
        alternatives = combined
    return alternatives


def compatibility(prior, cue_alternatives):
    results = set()
    for cues in cue_alternatives:
        for candidate in feature_alternatives(prior['head']):
            if any(name in candidate and candidate[name] != value for name, value in cues.items()):
                results.add('incompatible')
            elif cues and all(name in candidate for name in cues):
                results.add('compatible')
            else:
                results.add('unknown')
    return 'compatible' if 'compatible' in results else 'unknown' if 'unknown' in results else 'incompatible'


def extract(path, window, language):
    mentions, counts = read_corpus(path)
    histories = defaultdict(list)
    by_doc = defaultdict(list)
    for m in mentions:
        by_doc[m['doc']].append(m)
        histories[(m['doc'], m['meta']['eid'])].append(m)
    rows = []
    unresolved_rows = []
    construction_support = Counter()
    support = Counter()
    for m in mentions:
        n = m['head']
        links = subject_links(n)
        if not links:
            continue
        counts['identity_subject_mentions'] += 1
        if len(links) != 1:
            counts['excluded_multiple_subject_governors'] += 1
            continue
        pid, relation = links[0]
        if 'xsubj' in relation:
            counts['excluded_controlled_subjects'] += 1
            continue
        nodes = n['sentence']['nodes']
        predicate = nodes.get(pid)
        construction, reason = finite_construction(predicate, nodes, language)
        if construction is None:
            counts['excluded_' + reason] += 1
            continue
        person_status, person_evidence = third_person_evidence(construction, n, language)
        raw_form = 'null' if n['empty'] else n['upos'].lower()
        construction_support[(construction['kind'], person_status, raw_form)] += 1
        if person_status == 'not_confirmed_third':
            counts['excluded_existing_person_annotation_not_3'] += 1
            continue
        finite = construction['anchor']
        if n['feats'].get('Reflex') == 'Yes' or set(n['feats'].get('PronType', '').split(',')) & {'Rel', 'Int'}:
            counts['excluded_reflexive_relative_interrogative'] += 1
            continue
        if n['feats'].get('Person') not in (None, '3'):
            counts['excluded_subject_person_conflicts_with_third_person'] += 1
            continue
        if n['empty']:
            form = 'null'
        elif n['upos'] == 'PRON' and n['feats'].get('PronType') == 'Prs':
            if n['feats'].get('Case') and 'Nom' not in n['feats']['Case'].split(','):
                counts['excluded_nonnominative_pronominal_head'] += 1
                continue
            form = 'pronoun'
        elif n['upos'] in ('NOUN', 'PROPN'):
            form = 'np'
        else:
            counts['excluded_other_form'] += 1
            continue
        previous = [x for x in histories[(m['doc'], m['meta']['eid'])] if x['end'] < m['start']]
        if not previous:
            counts['excluded_no_prior_complete_mention'] += 1
            continue
        last = max(previous, key=lambda x: x['end'])
        recent = {}
        for x in by_doc[m['doc']]:
            if x['end'] >= m['start'] or x['meta']['eid'] == m['meta']['eid']:
                continue
            if n['sentence']['index'] - x['head']['sentence']['index'] > window:
                continue
            eid = x['meta']['eid']
            if eid not in recent or x['end'] > recent[eid]['end']:
                recent[eid] = x
        cues = construction_cues(construction)
        if not cues:
            counts['excluded_conflicting_construction_agreement'] += 1
            continue
        compatible = Counter(compatibility(x, cues) for x in recent.values())
        cue_type = ' OR '.join('|'.join(f'{k}={v}' for k, v in sorted(values.items())) for values in cues)
        row = {'source_file': str(path), 'doc_id': m['doc'], 'sent_id': n['sentence']['id'],
               'token_id': n['id'], 'entity_id': m['meta']['eid'], 'form': form,
               'subject_head_feats': n['feats'],
               'mention_text': ' '.join(w['form'] for w in m['words']),
               'predicate_id': predicate['id'], 'predicate_lemma': predicate['lemma'],
               'construction_type': construction['kind'],
               'construction_has_participle': any(x['feats'].get('VerbForm') == 'Part' for x in construction['cluster']),
               'construction_has_aux_or_cop': len(construction['cluster']) > 1,
               'construction_anchor_id': finite['id'], 'construction_anchor_form': finite['form'],
               'construction_anchor_feats': finite['feats'], 'construction_anchor_xpos': finite['xpos'],
               'person_status': person_status, 'person_evidence': person_evidence,
               'agreement_cue_sources': [dict(token_id=x['id'], form=x['form'], feats=x['feats'], xpos=x['xpos'])
                                         for x in construction['cue_nodes']],
               'annotation_cue_type': cue_type, 'annotation_cue_alternatives': cues,
               'mention_discontinuous': m['discontinuous'],
               'mention_node_ids': [f"{w['sentence']['id']}/{w['id']}" for w in m['words']],
               'prior_mention_discontinuous': last['discontinuous'],
               'entity_has_discontinuous_mentions': m['entity_has_discontinuous_mentions'],
               'recent_discontinuous_mentions': sum(x['discontinuous'] for x in recent.values()),
               'surface_disambiguation': 'unverified', 'subject_relation': relation,
               'prior_mentions': len(previous),
               'prior_distance_positions': m['start'] - last['end'],
               'prior_distance_sentences': n['sentence']['index'] - last['head']['sentence']['index'],
               'prior_target_feats': last['head']['feats'],
               'prior_target_type': last['meta'].get('etype', ''),
               'prior_target_subject': bool(subject_links(last['head'])),
               'recent_other_entities': len(recent),
               'compatible_entities': compatible['compatible'],
               'incompatible_entities': compatible['incompatible'],
               'unknown_compatibility_entities': compatible['unknown'],
               'opportunity_status': 'candidate_resource_specific_grammar_filter_pending'}
        (rows if person_status == 'confirmed_third' else unresolved_rows).append(row)
    # A position is ambiguous even if one mention has unresolved person.
    all_rows = rows + unresolved_rows
    positions = Counter((r['doc_id'], r['sent_id'], r['token_id']) for r in all_rows)
    counts['excluded_ambiguous_subject_positions'] = sum(n > 1 for n in positions.values())
    counts['excluded_candidates_at_ambiguous_positions'] = sum(n for n in positions.values() if n > 1)
    rows = [r for r in rows if positions[(r['doc_id'], r['sent_id'], r['token_id'])] == 1]
    unresolved_rows = [r for r in unresolved_rows if positions[(r['doc_id'], r['sent_id'], r['token_id'])] == 1]
    support.update((r['form'], r['annotation_cue_type']) for r in rows)
    counts['candidate_subjects_with_prior_mention'] = len(rows)
    counts['unresolved_person_candidates_with_prior_mention'] = len(unresolved_rows)
    summary = {'source_file': str(path), 'scope': 'feasibility_candidates_only',
               'coreference_annotation_status': 'present' if counts['entity_token_rows'] else 'absent_not_estimable',
               'protocol': 'finite_constructions_udapi_v3', 'language': language,
               'mention_parser': 'official UDAPI 682e4f69c9e01b2f9385d1d21bc66dc78539da31; strict coref loader',
               'competition_window_sentences': window, 'counts': dict(counts),
               'form_counts': dict(Counter(r['form'] for r in rows)),
               'unresolved_person_form_counts': dict(Counter(r['form'] for r in unresolved_rows)),
               'construction_support_before_form_and_prior_filters': [
                   dict(construction=c, person_status=p, raw_subject_form=f, count=n)
                   for (c, p, f), n in sorted(construction_support.items())],
               'confirmed_construction_form_counts': [dict(construction=c, form=f, count=n)
                   for (c, f), n in sorted(Counter((r['construction_type'], r['form']) for r in rows).items())],
               'person_evidence_counts': dict(Counter(r['person_evidence'] for r in rows)),
               'confirmed_finite_form_profile': [dict(construction=c, has_participle=p, has_aux_or_cop=a, form=f, count=n)
                   for (c, p, a, f), n in sorted(Counter(
                       (r['construction_type'], r['construction_has_participle'], r['construction_has_aux_or_cop'], r['form'])
                       for r in rows).items())],
               'unresolved_construction_form_counts': [dict(construction=c, form=f, count=n)
                   for (c, f), n in sorted(Counter((r['construction_type'], r['form']) for r in unresolved_rows).items())],
               'form_by_annotation_cue': [dict(form=f, cue_type=c, count=n)
                                          for (f, c), n in sorted(support.items())],
               'candidates_with_any_known_compatible_entity': sum(r['compatible_entities'] > 0 for r in rows),
               'candidates_with_unknown_competitor_features': sum(r['unknown_compatibility_entities'] > 0 for r in rows),
               'not_established': ['resource_specific_free_subject_opportunity_set',
                                   'form_independent_person_identification_for_czech_past',
                                   'surface_form_syncretism', 'model_or_theory_result']}
    return rows, unresolved_rows, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('conllu', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--language', choices=['cs', 'es', 'ca'], required=True)
    parser.add_argument('--window', type=int, default=2)
    args = parser.parse_args()
    rows, unresolved_rows, summary = extract(args.conllu, args.window, args.language)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'candidates.jsonl').open('w', encoding='utf-8') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    with (args.output / 'unresolved_person_candidates.jsonl').open('w', encoding='utf-8') as out:
        for row in unresolved_rows:
            out.write(json.dumps(row, ensure_ascii=False) + '\n')
    (args.output / 'inventory.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
