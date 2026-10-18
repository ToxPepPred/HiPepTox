#!/usr/bin/env python
"""Prepare the original StrucToxNet-style DSSP table and CB/CA contact records.

DSSP column offsets and contact computation follow the retained preprocessing
reference by Jiao Shihu. This script consumes PDB files; it does not predict them.
"""
import argparse
import csv
import importlib.util
from pathlib import Path
import shutil
import subprocess


def convert_dssp(source, destination, expected_length):
    column_ranges = [(35, 38), (85, 91), (91, 97), (97, 103), (103, 109),
                     (109, 115), (115, 122), (122, 129), (129, 136)]
    records = []
    seen_header = False
    for line in source.read_text().splitlines():
        if not seen_header:
            if line[2:3] == '#':
                seen_header = True
            continue
        if len(line) < 136 or not line[5:10].strip():
            continue
        number = int(line[5:10].strip())
        values = [line[start:end].strip() for start, end in column_ranges]
        if any(not value for value in values):
            raise ValueError('Incomplete DSSP record for residue %s' % number)
        for value in values:
            float(value)
        records.append([str(number), line[13], line[16].strip() or 'C'] + values)
    if not seen_header:
        raise ValueError('Expected classic fixed-width DSSP output, not mmCIF: ' + str(source))
    numbers = [int(record[0]) for record in records]
    if numbers != list(range(1, expected_length + 1)):
        raise ValueError('DSSP residue numbering/length differs from CSV sequence: ' + str(source))
    with destination.open('w', encoding='utf-8') as handle:
        handle.write('res\taa\tss\tsa\ttco\tkpa\talp\tphi\tpsi\tx\ty\tz\n')
        for record in records:
            handle.write('\t'.join(record) + '\n')


def run_dssp(executable, pdb, output):
    commands = [
        [executable, '--output-format=dssp', str(pdb), str(output)],
        [executable, '-i', str(pdb), '-o', str(output)],
    ]
    errors = []
    for command in commands:
        if output.exists():
            output.unlink()
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0 and output.is_file():
            return
        errors.append(' '.join(command) + ': ' + result.stderr[-1500:])
    raise RuntimeError('DSSP failed for %s\n%s' % (pdb, '\n'.join(errors)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--pdb-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--mkdssp', default='mkdssp')
    parser.add_argument('--cache-id-col', default='structoxnet_sample_id')
    parser.add_argument('--pdb-name-col', default='structoxnet_sample_id')
    parser.add_argument('--sequence-col', default='sequence')
    args = parser.parse_args()
    executable = shutil.which(args.mkdssp)
    if not executable:
        raise FileNotFoundError('DSSP executable not found: ' + args.mkdssp)
    reference_path = Path(__file__).resolve().with_name('structure_preprocessing_reference.py')
    spec = importlib.util.spec_from_file_location('structure_preprocessing_reference', reference_path)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    output = Path(args.output_dir).resolve()
    for folder in ['pdb', 'raw_dssp', 'dssp', 'pdb2rr']:
        (output / folder).mkdir(parents=True, exist_ok=True)
    seen = {}
    with open(args.input_csv, encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {args.cache_id_col, args.pdb_name_col, args.sequence_col}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing CSV columns: ' + repr(required - set(reader.fieldnames or [])))
        for row in reader:
            identifier = row[args.cache_id_col].strip()
            pdb_name = row[args.pdb_name_col].strip()
            sequence = row[args.sequence_col].strip().upper()
            for name in [identifier, pdb_name]:
                if not name or name in ['.', '..'] or '/' in name or '\\' in name:
                    raise ValueError('Invalid PDB identifier: ' + repr(name))
            if not sequence:
                raise ValueError('Empty peptide sequence: ' + identifier)
            if identifier in seen:
                if seen[identifier] != sequence:
                    raise ValueError('Conflicting sequences for ' + identifier)
                continue
            seen[identifier] = sequence
            source_pdb = (Path(args.pdb_dir) / (pdb_name + '.pdb')).resolve()
            if not source_pdb.is_file():
                raise FileNotFoundError(str(source_pdb))
            target_pdb = output / 'pdb' / (identifier + '.pdb')
            if source_pdb != target_pdb:
                shutil.copy2(source_pdb, target_pdb)
            raw_dssp = output / 'raw_dssp' / (identifier + '.dssp')
            run_dssp(executable, target_pdb, raw_dssp)
            convert_dssp(raw_dssp, output / 'dssp' / (identifier + '.dssp'), len(sequence))
            reference.cal_orientation(str(target_pdb), str(output / 'pdb2rr' / (identifier + '.pdb2rr')))
            print('Prepared structure inputs: ' + identifier, flush=True)


if __name__ == '__main__':
    main()
