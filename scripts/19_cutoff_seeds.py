import os
import sys
import json
import argparse
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get('GLOF_ROOT') or Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

MODELS_DIR = PROJECT_ROOT / 'models'
OUT = MODELS_DIR / 'cutoff_seeds.csv'
SCRIPTS = Path(__file__).resolve().parent
MIN_REGION_POS = 3


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def within_region(y, s, reg):
    from sklearn.metrics import roc_auc_score
    a = [roc_auc_score(y[reg == r], s[reg == r]) for r in np.unique(reg)
         if MIN_REGION_POS <= y[reg == r].sum() < (reg == r).sum()]
    return float(np.mean(a)) if a else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', type=int, default=10)
    ap.add_argument('--repeats', type=int, default=5)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    lt = load('lt', '08_lake_table.py')
    tv = load('tv', '10_train_validate.py')
    feats = json.loads((MODELS_DIR / 'feature_sets.json').read_text())['feature_sets']['full']
    labels = pd.read_csv(lt.LABELS).set_index('lake_phys')
    labels = labels[labels['label'] >= 0]
    info = lt.lake_info()
    ly = lt.lake_year_table()
    val = pd.read_csv(lt.PHYS_DIR / 'lake_year_validity.csv')
    model = tv.build_models(n_jobs=1)['BalancedRF']
    rows = []
    for k in range(args.seeds):
        lt.RANDOM_STATE = tv.RANDOM_STATE + 1000 + k
        lakes = lt.build_window(ly, val, labels, info, first_year=lt.DEM_YEAR)
        lakes = lakes[lakes['dem_before_event']].reset_index()
        X = lakes[feats].to_numpy(dtype=float)
        y = lakes['label'].to_numpy(dtype=int)
        oof, runs, _ = tv.run_scheme(model, X, y, lakes, 'group', args.repeats, None, args.jobs)
        s = np.nanmean(oof, axis=0)
        from sklearn.metrics import roc_auc_score
        rows.append({'seed': lt.RANDOM_STATE, 'n_lakes': len(y), 'n_pos': int(y.sum()),
                     'auc_mean_reps': float(runs['auc'].mean()),
                     'auc_pooled': float(roc_auc_score(y, s)),
                     'auc_within_region': within_region(y, s, lakes['analysis_region'].to_numpy())})
        print(f'  seed {lt.RANDOM_STATE}: lakes {len(y)} pos {int(y.sum())} '
              f'AUC {rows[-1]["auc_mean_reps"]:.3f} within {rows[-1]["auc_within_region"]:.3f}', flush=True)
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    print(f'AUC over {len(d)} cut-off draws: mean {d.auc_mean_reps.mean():.3f}, '
          f'range {d.auc_mean_reps.min():.3f}-{d.auc_mean_reps.max():.3f}; within-region mean '
          f'{d.auc_within_region.mean():.3f}, range {d.auc_within_region.min():.3f}-'
          f'{d.auc_within_region.max():.3f}; lakes {d.n_lakes.min()}-{d.n_lakes.max()}')
    print(f'saved {OUT.name}')


if __name__ == '__main__':
    main()
