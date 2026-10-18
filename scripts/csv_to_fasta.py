#!/usr/bin/env python
"""Export retained CSV sequences using the identifiers required by structure inputs."""
import argparse
import csv
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-csv', required=True)
    parser.add_argument('--output-fasta', required=True)
    parser.add_argument('--id-col', default='id')
    parser.add_argument('--sequence-col', default='sequence')
    args = parser.parse_args()
    records = {}
    with open(args.input_csv, encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        required = {args.id_col, args.sequence_col}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError('Missing CSV columns: ' + repr(required - set(reader.fieldnames or [])))
        for row in reader:
            identifier = row[args.id_col].strip()
            sequence = row[args.sequence_col].strip().upper()
            if not identifier or any(character.isspace() for character in identifier):
                raise ValueError('Invalid FASTA identifier: ' + repr(identifier))
            if not sequence or not sequence.isalpha() or not sequence.isascii():
                raise ValueError('Invalid peptide sequence for ' + identifier)
            if identifier in records and records[identifier] != sequence:
                raise ValueError('Conflicting sequences for identifier ' + identifier)
            records[identifier] = sequence
    output = Path(args.output_fasta)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8') as handle:
        for identifier, sequence in records.items():
            handle.write('>' + identifier + '\n' + sequence + '\n')
    print('Wrote %d sequences to %s' % (len(records), output))


if __name__ == '__main__':
    main()
