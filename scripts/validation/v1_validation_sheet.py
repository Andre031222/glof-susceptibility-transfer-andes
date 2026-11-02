import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get('GLOF_ROOT') or Path(__file__).resolve().parents[2])
MODELS_DIR = PROJECT_ROOT / 'models'
OOF = MODELS_DIR / 'oof_scores.csv'
LAKE_TABLE = PROJECT_ROOT / 'data' / 'processed' / 'labeled' / 'lake_table.csv'
OUT = PROJECT_ROOT / 'docs' / 'inventory_validation_sheet.csv'
RANDOM_STATE = 42
PER_AREA = 40
TOP_PCT = 2.0
N_DECILES = 10
PER_DECILE = 25


def oof_scores():
    for f in (OOF, LAKE_TABLE):
        if not f.exists():
            sys.exit(f'[ERROR] {f} not found; run 08_lake_table.py and 10_train_validate.py first')
    oof = pd.read_csv(OOF)
    lakes = (pd.read_csv(LAKE_TABLE, low_memory=False)
             .rename(columns={'lake_phys': 'lake_key', 'lake_elev_m': 'elev_mean'})
             [['lake_key', 'dist_glacier_m', 'elev_mean']])
    lakes = oof[['lake_key', 'analysis_region', 'lon', 'lat', 'area_m2', 'oof_score', 'glof']] \
        .merge(lakes, on='lake_key', how='inner')
    return lakes.rename(columns={'analysis_region': 'area_name', 'oof_score': 'oof',
                                 'glof': 'y'}).reset_index(drop=True)


def design_sample(lakes, rng):
    neg = lakes[lakes['y'] == 0]
    stratum = pd.qcut(neg['oof'].rank(method='first'), N_DECILES, labels=False) + 1
    lakes['design_stratum'] = stratum.reindex(lakes.index)
    lakes['inclusion_prob'] = np.nan
    lakes['in_design_sample'] = 0
    for d, g in neg.groupby(stratum):
        take = min(PER_DECILE, len(g))
        idx = rng.choice(g.index.values, size=take, replace=False)
        lakes.loc[idx, 'in_design_sample'] = 1
        lakes.loc[g.index, 'inclusion_prob'] = take / len(g)
    return lakes


def main():
    lakes = oof_scores()
    n = len(lakes)
    k = max(1, int(round(n * TOP_PCT / 100)))
    thr = lakes['oof'].sort_values(ascending=False).iloc[k - 1]
    lakes['in_watchlist_top2pct'] = (lakes['oof'] >= thr).astype(int)

    rng = np.random.RandomState(RANDOM_STATE)
    lakes = design_sample(lakes, rng)
    sample_idx = []
    for _, grp in lakes.groupby('area_name'):
        take = min(PER_AREA, len(grp))
        sample_idx.extend(rng.choice(grp.index.values, size=take, replace=False))
    wl = lakes[lakes.in_watchlist_top2pct == 1].index.values
    pos = lakes[lakes.y == 1].index.values
    design = lakes[lakes.in_design_sample == 1].index.values
    keep = sorted(set(sample_idx) | set(wl) | set(pos) | set(design))
    s = lakes.loc[keep].copy()

    s['google_earth_url'] = ('https://earth.google.com/web/@' + s['lat'].round(5).astype(str)
                             + ',' + s['lon'].round(5).astype(str) + ',1000a,3000d,35y,0h,0t,0r')
    s['gmaps_url'] = ('https://www.google.com/maps/@' + s['lat'].round(5).astype(str)
                      + ',' + s['lon'].round(5).astype(str) + ',2000m/data=!3m1!1e3')

    out = s[['lake_key', 'area_name', 'lat', 'lon', 'area_m2', 'dist_glacier_m',
             'elev_mean', 'oof', 'in_watchlist_top2pct', 'y', 'design_stratum',
             'inclusion_prob', 'in_design_sample', 'google_earth_url', 'gmaps_url']].copy()
    out = out.rename(columns={'oof': 'model_score', 'y': 'known_glof_source'})
    out['is_real_glacial_lake_YN'] = ''
    out['feature_type'] = ''
    out['confidence_1to3'] = ''
    out['notes'] = ''
    out = out.sort_values(['area_name', 'in_watchlist_top2pct', 'model_score'],
                          ascending=[True, False, False])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f'wrote {OUT}')
    print(f'  total lakes to review: {len(out)}')
    print(f'  watch-list lakes included: {int(out.in_watchlist_top2pct.sum())}')
    print(f'  known GLOF-source included: {int(out.known_glof_source.sum())}')
    print(f'  design-based sample (negatives, {PER_DECILE} per score decile, inclusion '
          f'probabilities recorded): {int(out.in_design_sample.sum())}')
    print('  per-area counts:')
    print(out.groupby('area_name').size().to_string())


if __name__ == '__main__':
    main()
