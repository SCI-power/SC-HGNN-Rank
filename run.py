"""Command-line entry point for the SC-HGNN-Rank study package."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'src'))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding='utf-8')


def environment(output, results=None):
    env = os.environ.copy()
    env['SC_HGNN_OUTPUT_DIR'] = str(output)
    env['SC_HGNN_RESULTS_DIR'] = str(results or ROOT / 'reference_results/model')
    env['SC_HGNN_ANALYSIS_OUTPUT'] = str(output)
    env['MPLBACKEND'] = 'Agg'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    return env


def execute(script, arguments, output, results=None, device=None):
    env = environment(output, results)
    if device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    command = [sys.executable, str(ROOT / script), *map(str, arguments)]
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, env=env, check=True)


def validate(output):
    manifest = ROOT / 'MANIFEST_SHA256.json'
    count = 0
    if manifest.exists():
        for record in json.loads(manifest.read_text(encoding='utf-8')):
            path = (ROOT / record['file']).resolve()
            if not path.is_relative_to(ROOT):
                raise ValueError('Manifest entry is outside the package.')
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != record['sha256']:
                raise ValueError('File checksum differs: ' + record['file'])
            count += 1
    execute('src/run_original_supplement.py', ['--phase', 'check'], output)
    write_json(output / 'package_validation.json', {'status': 'passed', 'files_checked': count})


def prepare(output):
    execute('src/build_spatial_causal_hgnn_graph_inputs.py', [], output)
    execute('src/build_spatial_causal_hgnn_tensor_dataset.py', [], output)
    import torch
    reference = torch.load(ROOT / 'data/model/spatial_causal_hgnn_tensor_dataset_2026-05-04.pt', map_location='cpu', weights_only=False)
    rebuilt = torch.load(output / 'preprocess/tensors/spatial_causal_hgnn_tensor_dataset_2026-05-04.pt', map_location='cpu', weights_only=False)
    differences = []
    def compare(a, b, name):
        if isinstance(a, torch.Tensor):
            if not isinstance(b, torch.Tensor) or not torch.equal(a, b):
                differences.append(name)
        elif isinstance(a, dict):
            if not isinstance(b, dict) or set(a) != set(b):
                differences.append(name + ':keys')
            for key in a.keys() & b.keys():
                compare(a[key], b[key], name + '.' + key)
        elif a != b:
            differences.append(name)
    compare(reference, rebuilt, 'tensor')
    write_json(output / 'preprocess/equivalence.json', {'exact_tensor_and_mapping_match': not differences, 'differences': differences})
    if differences:
        raise ValueError('Rebuilt graph differs from supplied tensors: ' + ', '.join(differences))


def infer(output, checkpoint=None):
    import numpy as np
    import pandas as pd
    import torch
    import reviewer_matched_rerun as core
    torch.set_num_threads(2)
    if checkpoint is None:
        record = json.loads((ROOT / 'checkpoints/index.json').read_text(encoding='utf-8'))[0]
        checkpoint = ROOT / record['path']
    checkpoint = checkpoint.resolve()
    raw = torch.load(ROOT / 'data/model/spatial_causal_hgnn_tensor_dataset_2026-05-04.pt', map_location='cpu', weights_only=False)
    labels = pd.read_csv(ROOT / 'data/model/Table_HGNN3_axis_label_table.csv')
    x, meta, _ = core.axis_features(labels)
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = core.TrainConfig(**state['config'])
    model = core.make_model(state['variant'], raw, x.shape[1], meta.shape[1], config)
    model.load_state_dict(state['state_dict'])
    model.eval()
    data = core.move_to_device(core.add_structural_features(raw), torch.device('cpu'))
    with torch.no_grad():
        _, score = core.forward_model(model, state['variant'], data, x, meta)
    columns = ['frozen_axis_id', 'cancer', 'compound_representative_name', 'target_gene_symbol', 'tf_gene_symbol', 'dominant_spatial_celltype']
    frame = labels[columns].copy()
    frame['predicted_score'] = score.cpu().numpy()
    frame['checkpoint_fold'] = state['split']['fold_id']
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / 'candidate_predictions.csv', index=False)
    comparison = checkpoint.parent / 'predictions.csv.gz'
    max_difference = None
    if comparison.exists():
        expected = pd.read_csv(comparison)[['frozen_axis_id', 'predicted_score']]
        joined = expected.merge(frame[['frozen_axis_id','predicted_score']], on='frozen_axis_id', suffixes=('_reference','_inference'), validate='one_to_one')
        max_difference = float(np.max(np.abs(joined.predicted_score_reference - joined.predicted_score_inference)))
        if max_difference > 1e-5:
            raise ValueError('Checkpoint inference differs from its saved test predictions.')
    write_json(output / 'inference_check.json', {'rows': len(frame), 'fold': state['split']['fold_id'],
        'max_abs_difference_from_saved_test_predictions': max_difference})
    print(f'Exported {len(frame)} predictions; reference difference={max_difference}', flush=True)


def scores(output):
    execute('src/recompute_available_external_scores.py', ['--output', output / 'external'], output)
    execute('src/recompute_priority_scores.py', ['--output', output / 'priority'], output)


def auxiliary(kind, mode, output, results, device, limit):
    folder = ROOT / 'analyses' / kind / 'scripts'
    if mode == 'summary':
        shutil.copytree(ROOT / 'reference_results' / kind / 'tables', output / 'tables', dirs_exist_ok=True)
        script = 'plot_panels.py' if kind == 'ablation' else 'summarize_and_draw.py'
        arguments = []
    elif mode == 'explain':
        if kind != 'ablation':
            raise ValueError('Explanation is available under the ablation command.')
        script, arguments = 'explain_frozen_model.py', ['--device', device if device != 'auto' else 'cpu']
    else:
        script = 'run_ablation.py' if kind == 'ablation' else 'run_graph_only.py'
        arguments = ['--check-only'] if mode == 'check' else []
        if limit is not None:
            arguments += ['--limit', str(limit)]
    execute(str((folder / script).relative_to(ROOT)), arguments, output, results, device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['validate','prepare','infer','scores','weights','smoke','train','summary','plot','ablation','graph-only'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--results', type=Path, help='Formal model-fit directory; defaults to packaged reference results.')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--device', choices=['auto','cpu','cuda'], default='cpu')
    parser.add_argument('--phase', choices=['main','sensitivity','missing','all'], default='all')
    parser.add_argument('--mode', choices=['check','train','summary','explain'], default='check')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    output = (args.output or ROOT / 'outputs' / args.command).resolve()
    protected = [ROOT / n for n in ['data','src','analyses','reference_results','checkpoints','environment','docs']]
    if output == ROOT or any(output == p or p in output.parents for p in protected):
        parser.error('Choose a separate output directory, not an input or source directory.')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.update(environment(output, args.results))
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    sys.dont_write_bytecode = True
    if args.command == 'validate':
        validate(output)
    elif args.command == 'prepare':
        prepare(output)
    elif args.command == 'infer':
        infer(output, args.checkpoint)
    elif args.command == 'scores':
        scores(output)
    elif args.command == 'weights':
        execute('src/reviewer_integrated_weight_sensitivity.py', [], output)
    elif args.command == 'smoke':
        started = time.perf_counter()
        validate(output / 'validation')
        prepare(output / 'rebuild')
        infer(output / 'inference')
        scores(output / 'scores')
        execute('src/run_original_supplement.py', ['--phase','smoke','--device',args.device], output / 'short_training', device=args.device)
        write_json(output / 'SMOKE_SUCCESS.json', {'status':'passed','short_training_models':8,'epochs':3,
            'seconds':round(time.perf_counter()-started,3), 'formal_results_overwritten':False})
    elif args.command == 'train':
        execute('src/run_original_supplement.py', ['--phase',args.phase,'--device',args.device], output, args.results, args.device)
    elif args.command == 'summary':
        execute('src/summarize_original_supplement.py', [], output, args.results)
    elif args.command == 'plot':
        execute('src/plot_original_supplement.py', [], output, args.results)
    else:
        auxiliary('graph_only' if args.command == 'graph-only' else 'ablation', args.mode, output, args.results, args.device, args.limit)


if __name__ == '__main__':
    main()
