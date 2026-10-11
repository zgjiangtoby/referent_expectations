#!/usr/bin/env python3
"""Prepare form-model rows and check covariate support."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

FEATURES = ['log_total', 'log_distance', 'log_prior_mentions', 'prior_subject',
            'current_plural', 'prior_plural', 'prior_number_unknown',
            'aux_no_part', 'has_part', 'clause_coord', 'clause_embedded']
CONTINUOUS = ['log_total', 'log_distance', 'log_prior_mentions']
SEED = 20261008


def jsonl(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def key(row):
    return row['doc_id'], row['sent_id'], row['token_id']


def predicate_relations(path, selected):
    result = {}
    sent = None
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            if line.startswith('# sent_id = '):
                sent = line.rstrip('\n').split(' = ', 1)[1]
            elif line.strip() and not line.startswith('#'):
                fields = line.rstrip('\n').split('\t')
                if (sent, fields[0]) in selected:
                    result[(sent, fields[0])] = fields[7]
    return result


def feature_matrix(rows, names):
    return np.array([[1.0, row['competition_c']] + [row[name] for name in names]
                     for row in rows], dtype=float)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for lang in ('ca', 'cs', 'es'):
        parser.add_argument('--' + lang + '-root', type=Path, required=True,
                            help='Measurement root containing per-file coverage directories')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roots = {lang: getattr(args, lang + '_root').resolve() for lang in ('ca', 'cs', 'es')}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for lang, root in roots.items():
        rows = []
        doc_files = defaultdict(set)
        raw_possessive_subjects = Counter()
        for coverage_path in sorted(root.glob('*/coverage.json')):
            coverage = json.loads(coverage_path.read_text())
            if coverage['coreference_annotation_status'] != 'present':
                continue
            split = coverage_path.parent.name
            raw_path = Path(coverage['source_file'])
            canonical = Path(coverage['canonical_v3'])
            reference = {key(row): row for row in jsonl(canonical / 'candidates.jsonl')}
            projected = {row['projection_id']: row for row in jsonl(coverage_path.parent / 'antecedent_features.jsonl')}
            candidates = [row for row in jsonl(coverage_path.parent / 'candidates.jsonl') if row['scope'] == 'primary']
            selected = {(row['sent_id'], reference[key(row)]['predicate_id']) for row in candidates}
            relations = predicate_relations(raw_path, selected)
            for row in candidates:
                ref = reference[key(row)]
                prior = projected[row['prior_target_projection_id']]['features']['Number']
                prior_number = prior['values'][0] if prior['status'] == 'known' and prior['values'][0] in ('Sing', 'Plur') else 'unknown'
                rel = relations[(row['sent_id'], ref['predicate_id'])]
                rel_base = rel.split(':')[0]
                clause = 'root' if rel_base == 'root' else 'coord' if rel_base in ('conj', 'parataxis') else 'embedded'
                match = row['projected_number']
                doc_files[row['doc_id']].add(split)
                if row['form'] == 'pronoun' and ref['subject_head_feats'].get('Poss') == 'Yes':
                    raw_possessive_subjects[split] += 1
                result = {'language': lang, 'split': split, 'doc_id': row['doc_id'],
                          'entity_id': row['entity_id'], 'sent_id': row['sent_id'], 'token_id': row['token_id'],
                          'form': row['form'], 'y': {'null': 0, 'pronoun': 1, 'np': 2}[row['form']],
                          'eligible': row['primary_model_eligible'],
                          'matching_lower': match['lower'], 'matching_upper': match['upper'],
                          'unknown': match['unknown'], 'ambiguous': match['ambiguous'],
                          'total_entities': row['recent_other_entities'],
                          'log_matching_lower': float(np.log1p(match['lower'])),
                          'log_matching_upper': float(np.log1p(match['upper'])),
                          'log_total': float(np.log1p(row['recent_other_entities'])),
                          'log_distance': float(np.log1p(row['prior_distance_positions'])),
                          'log_prior_mentions': float(np.log1p(row['prior_mentions'])),
                          'prior_subject': int(row['prior_target_subject']),
                          'current_plural': int(row['finite_number'] == 'Plur'),
                          'prior_plural': int(prior_number == 'Plur'),
                          'prior_number_unknown': int(prior_number == 'unknown'),
                          'aux_no_part': int(row['construction_has_aux_or_cop'] and not row['construction_has_participle']),
                          'has_part': int(row['construction_has_participle']),
                          'clause_coord': int(clause == 'coord'), 'clause_embedded': int(clause == 'embedded'),
                          'predicate_relation': rel, 'prior_number': prior_number}
                rows.append(result)
        eligible = [row for row in rows if row['eligible']]
        domain_max = max(row['log_matching_lower'] for row in eligible)
        docs = sorted({row['doc_id'] for row in eligible})
        permuted = np.random.default_rng(SEED).permutation(len(docs))
        folds = {docs[index]: i % 5 for i, index in enumerate(permuted)}
        for row in rows:
            row['competition_c'] = row['log_matching_lower'] / domain_max
            row['fold'] = folds.get(row['doc_id'])
        names = [name for name in FEATURES if len({row[name] for row in eligible}) > 1]
        matrix = feature_matrix(eligible, names)
        support = {name: {'min': min(row[name] for row in eligible), 'max': max(row[name] for row in eligible),
                          'distinct': len({row[name] for row in eligible})} for name in FEATURES}
        forms = []
        for sample, data in [('primary', rows), ('complete', eligible)]:
            for form in ('null', 'pronoun', 'np'):
                selected_rows = [row for row in data if row['form'] == form]
                forms.append({'sample': sample, 'form': form, 'rows': len(selected_rows),
                              'documents': len({row['doc_id'] for row in selected_rows}),
                              'entities': len({(row['split'], row['doc_id'], row['entity_id']) for row in selected_rows})})
        fold_support = []
        for fold in range(5):
            held = [row for row in eligible if row['fold'] == fold]
            train = [row for row in eligible if row['fold'] != fold]
            fold_support.append({'fold': fold, 'held_rows': len(held), 'held_documents': len({row['doc_id'] for row in held}),
                                 'held_forms': dict(Counter(row['form'] for row in held)),
                                 'train_forms': dict(Counter(row['form'] for row in train)),
                                 'train_rank': int(np.linalg.matrix_rank(feature_matrix(train, names)))})
        summary = {'language': lang, 'rows_primary': len(rows), 'rows_complete': len(eligible),
                   'forms': forms, 'features': names, 'constant_columns_removed': [name for name in FEATURES if name not in names],
                   'design_columns': matrix.shape[1], 'design_rank': int(np.linalg.matrix_rank(matrix)),
                   'feature_support': support, 'competition_domain_max_log': domain_max,
                   'document_fold_seed': SEED, 'fold_support': fold_support,
                   'document_ids_in_multiple_files': {doc: sorted(files) for doc, files in doc_files.items() if len(files)>1},
                   'current_possessive_pronouns_in_primary': dict(raw_possessive_subjects),
                   'field_limitations': ['distance_positions_includes_empty_nodes', 'no_annotated_animacy_focus_or_free_choice',
                                         'unknown_competition_not_identified_in_full_population'], 'effect_fitted': False}
        summaries[lang] = summary
        with (output / f'{lang}_rows.jsonl').open('w', encoding='utf-8') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + '\n')
        print(json.dumps({key: summary[key] for key in ('language', 'rows_complete', 'design_columns', 'design_rank',
              'constant_columns_removed', 'current_possessive_pronouns_in_primary', 'fold_support')}, ensure_ascii=False), flush=True)
    (output / 'design_support.json').write_text(json.dumps(summaries, ensure_ascii=False, indent=2) + '\n')


if __name__ == '__main__':
    main()
