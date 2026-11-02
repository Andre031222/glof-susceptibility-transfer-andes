import os
import sys
import json
import argparse
import warnings
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

PROJECT_ROOT = Path(os.environ.get('GLOF_ROOT') or Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

MODELS_DIR = PROJECT_ROOT / 'models'
LAKE_TABLE = PROJECT_ROOT / 'data' / 'processed' / 'labeled' / 'lake_table.csv'
OUT = MODELS_DIR / 'learning_curve.csv'
FRACTIONS = [0.2, 0.35, 0.5, 0.65, 0.8, 1.0]
MIN_REGION_POS = 3
RANDOM_STATE = 42

spec = importlib.util.spec_from_file_location('tv', Path(__file__).resolve().parent /
                                              '10_train_validate.py')
tv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tv)


def one_rep(model, X, y, groups, frac, rep):
    from sklearn.model_selection import StratifiedGroupKFold
    rng = np.random.default_rng(RANDOM_STATE + 1000 * rep + int(100 * frac))
    oof = np.full(len(y), np.nan)
    n_pos_train = []
    split = StratifiedGroupKFold(tv.N_SPLITS, shuffle=True, random_state=RANDOM_STATE + rep)
    for tr, te in split.split(X, y, groups):
        pos = tr[y[tr] == 1]
        k = max(2, int(round(frac * len(pos))))
        keep = np.concatenate([tr[y[tr] == 0], rng.choice(pos, size=min(k, len(pos)),
                                                          replace=False)])
        n_pos_train.append(int(y[keep].sum()))
        oof[te] = tv.fit_predict(model, X[keep], y[keep], X[te])
    return frac, rep, oof, float(np.mean(n_pos_train))


def scores(y, s, reg):
    from sklearn.metrics import roc_auc_score, average_precision_score
    ok = np.isfinite(s)
    within = [roc_auc_score(y[ok & (reg == r)], s[ok & (reg == r)]) for r in np.unique(reg)
              if y[ok & (reg == r)].sum() >= MIN_REGION_POS]
    return (float(roc_auc_score(y[ok], s[ok])), float(average_precision_score(y[ok], s[ok])),
            float(np.mean(within)) if within else np.nan)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--repeats', type=int, default=20)
    ap.add_argument('--jobs', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = ap.parse_args()
    cfg = json.loads((MODELS_DIR / 'feature_sets.json').read_text())
    feats = cfg['feature_sets']['full']
    lakes = pd.read_csv(LAKE_TABLE, low_memory=False)
    lakes = lakes[lakes['dem_before_event']].reset_index(drop=True)
    X = lakes[feats].to_numpy(dtype=float)
    y = lakes['label'].to_numpy(dtype=int)
    groups = lakes['group'].astype(str).to_numpy()
    reg = lakes['analysis_region'].to_numpy()
    model = tv.build_models(n_jobs=1)['BalancedRF']
    print(f'=== learning curve: lakes={len(y)} positives={int(y.sum())} '
          f'fractions={FRACTIONS} repeats={args.repeats}')
    out = Parallel(n_jobs=args.jobs)(delayed(one_rep)(model, X, y, groups, f, r)
                                     for f in FRACTIONS for r in range(args.repeats))
    rows = []
    for frac, rep, s, npos in out:
        auc, pr, within = scores(y, s, reg)
        rows.append({'fraction': frac, 'rep': rep, 'n_pos_train_mean': npos, 'auc': auc,
                     'pr_auc': pr, 'auc_within_region': within})
    d = pd.DataFrame(rows)
    d.to_csv(OUT, index=False)
    summ = d.groupby('fraction').agg(n_pos_train=('n_pos_train_mean', 'mean'),
                                     auc=('auc', 'mean'), auc_lo=('auc', lambda v: v.quantile(0.025)),
                                     auc_hi=('auc', lambda v: v.quantile(0.975)),
                                     within=('auc_within_region', 'mean'))
    print(summ.round(3).to_string())
    print(f'saved {OUT.name}')


if __name__ == '__main__':
    main()
