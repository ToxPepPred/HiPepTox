from pathlib import Path
from datetime import datetime
import csv
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time

import numpy
import pandas
import sklearn
import xgboost

BASE = Path('/home/cns/HiPepTox/Paper')
ROOT = BASE / 'model/baseline/toxteller'
OUT = ROOT / 'results/full_length_5_50_20261013'
SEEDS = [42, 123, 2024]
MODELS = ['lr', 'rf', 'svm', 'xgboost']
EXPECTED = {'0.8': {'train': 5544, 'test': 1388}, '0.9': {'train': 7056, 'test': 1766}}
START = time.time()

def save_json(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)

def progress(**extra):
    data = {'updated_at': datetime.now().isoformat(), 'elapsed_seconds': time.time() - START, 'total_runs': 6, 'total_model_fits': 24}
    data.update(extra)
    save_json(OUT / 'progress.json', data)

def metric_from_rows(rows, model):
    pairs = [(int(r['label']), int(r[model + '_pred'])) for r in rows]
    tp = sum(y == 1 and p == 1 for y, p in pairs)
    tn = sum(y == 0 and p == 0 for y, p in pairs)
    fp = sum(y == 0 and p == 1 for y, p in pairs)
    fn = sum(y == 1 and p == 0 for y, p in pairs)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return {'accuracy': (tp + tn) / len(rows), 'sensitivity': tp / (tp + fn), 'specificity': tn / (tn + fp), 'mcc': (tp * tn - fp * fn) / den if den else 0.0, 'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn}

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'full_length_summary.json').exists():
        raise RuntimeError('Completed results already exist; preserve them instead of overwriting.')
    code = OUT / 'code'
    code.mkdir(exist_ok=True)
    sources = {}
    for name in ['train_eval_csv.py', 'extractfeature.py']:
        source = ROOT / 'program_resource' / name
        destination = code / name
        shutil.copy2(source, destination)
        sources[name] = {'source': str(source), 'snapshot': str(destination), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest()}
    manifest = {'benchmark_variant': 'adapted_ToxTeller_5_50_aa', 'terminal_encoding': '8-position N/C profiles; unavailable positions are zero padded', 'seeds': SEEDS, 'models': MODELS, 'n_jobs': 4, 'python': sys.executable, 'software': {'python': platform.python_version(), 'numpy': numpy.__version__, 'pandas': pandas.__version__, 'sklearn': sklearn.__version__, 'xgboost': xgboost.__version__}, 'sources': sources, 'datasets': {}, 'model_selection': 'All four classifier configurations and all three seeds are prespecified and reported; no test-driven tuning.'}
    for protocol in EXPECTED:
        tag = '08' if protocol == '0.8' else '09'
        train = BASE / ('data/processed/toxipep%s_internal_85_15/peptide_toxicity.csv' % tag)
        test = BASE / ('data/processed/toxipep%s_external_test/peptide_toxicity.csv' % tag)
        datasets = {}
        for role, path in [('train', train), ('test', test)]:
            with path.open(newline='') as f:
                rows = list(csv.DictReader(f))
            if len(rows) != EXPECTED[protocol][role]:
                raise RuntimeError('Unexpected full dataset size: ' + str(path))
            lengths = [len(r['sequence'].strip()) for r in rows]
            positive = sum(int(r['toxicity']) == 1 for r in rows)
            datasets[role] = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'n': len(rows), 'positive': positive, 'negative': len(rows) - positive, 'min_length': min(lengths), 'max_length': max(lengths), 'under_10_aa': sum(n < 10 for n in lengths), 'under_8_aa': sum(n < 8 for n in lengths)}
        manifest['datasets'][protocol] = datasets
    save_json(OUT / 'run_manifest.json', manifest)

    env = dict(os.environ, OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4', NUMEXPR_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    runs = []
    for protocol in ['0.8', '0.9']:
        for seed in SEEDS:
            name = 'cdhit%s_seed%s' % (protocol.replace('.', ''), seed)
            output = OUT / name
            log = OUT / (name + '.log')
            progress(phase='training', dataset=protocol, seed=seed, completed_runs=len(runs), log=str(log))
            command = [sys.executable, str(code / 'train_eval_csv.py'), '--train_csv', manifest['datasets'][protocol]['train']['path'], '--test_csv', manifest['datasets'][protocol]['test']['path'], '--output_dir', str(output), '--min_len', '5', '--max_len', '50', '--random_state', str(seed), '--n_jobs', '4']
            run_start = time.time()
            with log.open('w') as handle:
                result = subprocess.run(command, cwd=str(code), env=env, stdout=handle, stderr=subprocess.STDOUT)
            if result.returncode:
                progress(phase='failed', dataset=protocol, seed=seed, completed_runs=len(runs), returncode=result.returncode, log=str(log))
                raise RuntimeError('Training failed; see ' + str(log))
            summary = json.loads((output / 'train_eval_summary.json').read_text())
            for role, key in [('train', 'train_dataset'), ('test', 'external_test_dataset')]:
                data = summary[key]
                if data['kept_rows'] != EXPECTED[protocol][role] or data['filtered_invalid_chars'] or data['filtered_out_of_length_range']:
                    raise RuntimeError('Full dataset coverage was not achieved.')
            with (output / 'external_test_predictions.csv').open(newline='') as handle:
                prediction_rows = list(csv.DictReader(handle))
            if len(prediction_rows) != EXPECTED[protocol]['test']:
                raise RuntimeError('Prediction count differs from full test set.')
            verified = {}
            for model in MODELS:
                calculated = metric_from_rows(prediction_rows, model)
                recorded = summary['models'][model]['external_test']
                for metric in ['accuracy', 'sensitivity', 'specificity', 'mcc']:
                    if abs(calculated[metric] - recorded[metric]) > 1e-12:
                        raise RuntimeError('Prediction/metric mismatch for ' + model)
                if abs(calculated['accuracy'] - (calculated['sensitivity'] + calculated['specificity']) / 2) > 1e-12:
                    raise RuntimeError('Balanced-test metric identity failed.')
                verified[model] = calculated
            runs.append({'dataset': protocol, 'seed': seed, 'directory': str(output), 'predictions': str(output / 'external_test_predictions.csv'), 'summary': str(output / 'train_eval_summary.json'), 'elapsed_seconds': time.time() - run_start, 'metrics': verified})
            progress(phase='training', completed_runs=len(runs), last_completed=name)

    aggregate = []
    for protocol in ['0.8', '0.9']:
        for model in MODELS:
            records = [r['metrics'][model] for r in runs if r['dataset'] == protocol]
            item = {'dataset': protocol, 'model': model, 'n_test': EXPECTED[protocol]['test'], 'n_runs': 3}
            for metric in ['accuracy', 'sensitivity', 'specificity', 'mcc']:
                values = [r[metric] for r in records]
                item[metric + '_mean'] = statistics.mean(values)
                item[metric + '_sample_sd'] = statistics.stdev(values)
            aggregate.append(item)
    final = {'manifest': manifest, 'runs': runs, 'aggregate': aggregate, 'elapsed_seconds': time.time() - START, 'all_full_test_samples_predicted': True, 'all_balanced_metric_identities_verified': True}
    save_json(OUT / 'full_length_summary.json', final)
    with (OUT / 'full_length_metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate[0]))
        writer.writeheader()
        writer.writerows(aggregate)
    lines = ['# Adapted ToxTeller: full 5-50 aa benchmark', '', '| Dataset | Model | Test n | Accuracy | Sensitivity | Specificity | MCC |', '| --- | --- | ---: | --- | --- | --- | --- |']
    for row in aggregate:
        metrics = ['%.4f +/- %.4f' % (row[m + '_mean'], row[m + '_sample_sd']) for m in ['accuracy', 'sensitivity', 'specificity', 'mcc']]
        lines.append('| %s | %s | %s | %s |' % (row['dataset'], row['model'].upper(), row['n_test'], ' | '.join(metrics)))
    lines += ['', 'Three seeds: 42, 123, 2024. Fixed original classifier hyperparameters; full historical training pools.', 'Expanded input range with zero-padding of missing terminal positions. This is an adapted benchmark implementation, extending the original 10-50 aa scope.', 'Historical results and source backups are preserved. See run_manifest.json for versions, hashes, inputs and settings.']
    (OUT / 'full_length_results.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    progress(phase='completed', completed_runs=6, completed_model_fits=24, results=str(OUT / 'full_length_results.md'))
    print(json.dumps({'status': 'completed', 'result_directory': str(OUT), 'elapsed_seconds': final['elapsed_seconds']}), flush=True)

if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            progress(phase='failed', error=repr(exc))
        raise
