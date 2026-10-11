"""Prepare the fixed CorefUD files for referent and form measurements."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

CODE = Path(__file__).resolve().parent

FILES = {'es': [('es_ancora-ud-train.conllu', 43100702, True),
        ('es_ancora-ud-dev.conllu', 5121417, True),
        ('es_ancora-ud-test.conllu', 5123220, True)],
 'ca': [('ca_ancora-ud-train.conllu', 39855232, True),
        ('ca_ancora-ud-dev.conllu', 5441926, True),
        ('ca_ancora-ud-test.conllu', 5534602, True)],
 'cs': [('cs_pdt-ud-train-ca.conllu', 8545136, False),
        ('cs_pdt-ud-train-ct.conllu', 8184437, True),
        ('cs_pdt-ud-train-la.conllu', 24128843, False),
        ('cs_pdt-ud-train-lt.conllu', 68655552, True),
        ('cs_pdt-ud-train-ma.conllu', 9966661, False),
        ('cs_pdt-ud-train-mt.conllu', 12694593, True),
        ('cs_pdt-ud-train-va.conllu', 14462447, False),
        ('cs_pdt-ud-dev.conllu', 20019876, True),
        ('cs_pdt-ud-test.conllu', 21633518, True)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for language in FILES:
        parser.add_argument("--" + language + "-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Use a new output directory")
    inputs = []
    for language, entries in FILES.items():
        data = getattr(args, language + "_data").resolve()
        for name, size, annotated in entries:
            path = data / name
            if not path.is_file() or path.stat().st_size != size:
                parser.error("Missing or size-mismatched source: " + str(path))
            inputs.append((language, path, annotated))
    output = args.output.resolve()
    output.mkdir(parents=True)
    for language, path, annotated in inputs:
        reference = output / "reference" / language / path.stem
        measurement = output / "measurement" / language / path.stem
        subprocess.run([sys.executable, str(CODE / "extract_candidates.py"), str(path),
            "--language", language, "--window", "2", "--output", str(reference)], check=True)
        inventory = json.loads((reference / "inventory.json").read_text())
        if (inventory["coreference_annotation_status"] == "present") != annotated:
            raise ValueError("Unexpected coreference layer: " + path.name)
        subprocess.run([sys.executable, str(CODE / "measure_features.py"), "--canonical", str(reference),
            "--output", str(measurement)], check=True)
    subprocess.run([sys.executable, str(CODE / "prepare_analysis.py"),
        "--ca-root", str(output / "measurement/ca"), "--cs-root", str(output / "measurement/cs"),
        "--es-root", str(output / "measurement/es"), "--output", str(output / "analysis")], check=True)


if __name__ == "__main__":
    main()
