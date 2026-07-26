import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from eda_toolkit import analyze, run_advanced_pipeline
from eda_toolkit import load_csv

DATASETS = [
    {
        'name': 'd1',
        'path': os.path.join('..', 'd1.csv'),
        'target': 'Survived',
        'outdir': os.path.join('..', 'eda_reports', 'd1'),
        'run_pipeline': False,
    },
    {
        'name': 'd2',
        'path': os.path.join('..', 'd2.csv'),
        'target': 'Species',
        'outdir': os.path.join('..', 'eda_reports', 'd2'),
        'run_pipeline': True,
    },
    {
        'name': 'd3',
        'path': os.path.join('..', 'd3.csv'),
        'target': 'AQI Category',
        'outdir': os.path.join('..', 'eda_reports', 'd3'),
        'run_pipeline': False,
    },
]


def run_dataset(cfg, one_hot_thresh: int, cv: int, run_pipeline: bool):
    dataset_name = cfg['name']
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), cfg['path']))
    outdir = os.path.abspath(os.path.join(os.path.dirname(__file__), cfg['outdir']))

    print(f"\n=== {dataset_name.upper()} ===")
    print(f"Loading {csv_path}")
    df = load_csv(csv_path)
    os.makedirs(outdir, exist_ok=True)

    print(f"Running generic EDA for {dataset_name} with target '{cfg['target']}'")
    try:
        results = analyze(df, cfg['target'], outdir=outdir, run_baseline=True)
        print(f"EDA done. Report saved to {os.path.join(outdir, 'report.md')}")
        print(f"Target type: {results.get('target_type')}  |  features analyzed: {len(results.get('features', {}))}")
    except Exception as exc:
        print(f"FAILED generic EDA for {dataset_name}: {exc}")

    if run_pipeline and cfg.get('run_pipeline', False):
        print(f"Running advanced pipeline for {dataset_name}")
        try:
            pipeline_results = run_advanced_pipeline(df, target=cfg['target'], outdir=outdir,
                                                    one_hot_thresh=one_hot_thresh, cv=cv)
            print(f"Advanced pipeline done. Best score: {pipeline_results.get('best_score')}")
        except Exception as exc:
            print(f"FAILED advanced pipeline for {dataset_name}: {exc}")


def main():
    parser = argparse.ArgumentParser(description='Run generic EDA for all known datasets and advanced pipeline for d2')
    parser.add_argument('--one-hot-thresh', type=int, default=3,
                        help='Low-cardinality threshold for one-hot encoding in the advanced pipeline')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of CV folds for the advanced pipeline')
    parser.add_argument('--datasets', nargs='*', choices=[cfg['name'] for cfg in DATASETS],
                        default=[cfg['name'] for cfg in DATASETS],
                        help='Datasets to run (default: all)')
    parser.add_argument('--skip-pipeline', action='store_true',
                        help='Skip the advanced pipeline step even for d2')
    args = parser.parse_args()

    selected = [cfg for cfg in DATASETS if cfg['name'] in args.datasets]
    if not selected:
        raise SystemExit('No datasets selected.')

    for cfg in selected:
        run_dataset(cfg, one_hot_thresh=args.one_hot_thresh, cv=args.cv,
                    run_pipeline=not args.skip_pipeline)

    print('\nAll requested datasets processed.')


if __name__ == '__main__':
    main()
