import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get('GLOF_ROOT') or Path(__file__).resolve().parent.parent)
MODELS_DIR = PROJECT_ROOT / 'models'
LAKE_TABLE = PROJECT_ROOT / 'data' / 'processed' / 'labeled' / 'lake_table.csv'
OUT = MODELS_DIR / 'inventory_screen.csv'


def load_lakes():
    for f in (LAKE_TABLE, MODELS_DIR / 'oof_scores.csv'):
        if not f.exists():
            sys.exit(f'[ERROR] {f} not found; run 08_lake_table.py and 10_train_validate.py first')
    oof = pd.read_csv(MODELS_DIR / 'oof_scores.csv')
    lakes = pd.read_csv(LAKE_TABLE, low_memory=False).set_index('lake_phys')
    lakes = lakes.reindex(oof['lake_key'])
    if lakes['label'].isna().any():
        raise ValueError('oof_scores.csv and lake_table.csv list different lakes; re-run 10')
    lakes = lakes.reset_index(drop=True)
    lakes['oof'] = oof['oof_score'].values
    lakes['y'] = oof['glof'].values.astype(int)
    return lakes


def main():
    lakes = load_lakes()
    n = len(lakes)
    pos = lakes[lakes.y == 1]

    d = lakes['dist_glacier_m']
    e = lakes['lake_elev_m']
    dp, ep = pos['dist_glacier_m'], pos['lake_elev_m']

    d_thr = float(dp.quantile(0.95))
    e_lo = float(ep.quantile(0.05))
    shape_thr = float(lakes['elongation'].quantile(0.99)) if 'elongation' in lakes else np.inf

    off_ice = d > d_thr
    off_elev = e < e_lo
    off_shape = (lakes['elongation'] > shape_thr if 'elongation' in lakes
                 else pd.Series(False, index=lakes.index))
    off_context = off_ice | off_elev
    missing_context = (d.isna() | e.isna()) & ~off_context
    in_context = ~off_context & ~missing_context

    print(f'=== glacial-context envelope (from {len(pos)} GLOF-source lakes) ===')
    print(f'  dist_glacier_m: pos median {dp.median():.0f}, pos p95 {d_thr:.0f}, '
          f'pop median {d.median():.0f}')
    print(f'  lake_elev_m: pos median {ep.median():.0f}, pos p05 {e_lo:.0f}, '
          f'pop median {e.median():.0f}')
    print(f'  elongation p99 threshold: {shape_thr:.2f}')
    print()
    print(f'=== modelled inventory (n={n}) ===')
    print(f'  off-ice (>{d_thr:.0f} m from glacier): {int(off_ice.sum())} '
          f'({100 * off_ice.mean():.1f}%)')
    print(f'  off-elev (<{e_lo:.0f} m): {int(off_elev.sum())} ({100 * off_elev.mean():.1f}%)')
    print(f'  off-context (either): {int(off_context.sum())} ({100 * off_context.mean():.1f}%)')
    print(f'  context unknown (dist_glacier_m or lake_elev_m missing, not off-context): '
          f'{int(missing_context.sum())} ({100 * missing_context.mean():.1f}%)')
    print(f'  extreme-elongation flag: {int(off_shape.sum())} ({100 * off_shape.mean():.1f}%)')
    print(f'  positives off-context (5-10% by construction of the envelope): '
          f'{int(off_context[lakes.y == 1].sum())} of {len(pos)}; context unknown '
          f'{int(missing_context[lakes.y == 1].sum())}')
    print()
    print(f'  median OOF score in-context : {lakes.loc[in_context, "oof"].median():.3f}')
    print(f'  median OOF score off-context: {lakes.loc[off_context, "oof"].median():.3f}')
    print(f'  median OOF score unknown    : {lakes.loc[missing_context, "oof"].median():.3f}')
    print()

    order = lakes['oof'].fillna(0).sort_values(ascending=False)
    rows = []
    for pct in (0.5, 1.0, 2.0, 5.0):
        k = max(1, int(round(n * pct / 100)))
        idx = order.index[:k]
        sub = lakes.loc[idx]
        oc = off_context.loc[idx]
        mc = missing_context.loc[idx]
        ic = in_context.loc[idx]
        rows.append({'top_pct': pct, 'n_lakes': k,
                     'off_context': int(oc.sum()),
                     'off_context_pct': round(100 * oc.mean(), 1),
                     'missing_context': int(mc.sum()),
                     'missing_context_pct': round(100 * mc.mean(), 1),
                     'in_context': int(ic.sum()),
                     'pos_in_set': int(sub.y.sum())})
        print(f'=== top {pct}% watch-list (n={k}) ===')
        print(f'  off-context (likely commission / non-glacial): {int(oc.sum())} '
              f'({100 * oc.mean():.1f}%)')
        print(f'  context unknown (missing glacier distance or elevation): {int(mc.sum())} '
              f'({100 * mc.mean():.1f}%)')
        print(f'  in plausible glacial context: {int(ic.sum())} ({100 * ic.mean():.1f}%)')

    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f'\nwrote {OUT}')


if __name__ == '__main__':
    main()
