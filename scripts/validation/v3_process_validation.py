import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get('GLOF_ROOT') or Path(__file__).resolve().parents[2])
DEFAULT_IN = PROJECT_ROOT / 'docs' / 'inventory_validation_results.csv'
DEFAULT_SHEET = PROJECT_ROOT / 'docs' / 'inventory_validation_sheet.csv'
DEFAULT_OOF = PROJECT_ROOT / 'models' / 'oof_scores.csv'
DEFAULT_OUT = PROJECT_ROOT / 'models'
TOPK_PCT = [0.5, 1.0, 2.0, 5.0, 10.0]
Z = 1.96


def wilson(k, n, z=Z):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def yn(v):
    s = str(v).strip().upper()
    if s in ('Y', 'YES', 'S', 'SI', 'TRUE', '1', 'T'):
        return 1.0
    if s in ('N', 'NO', 'FALSE', '0', 'F'):
        return 0.0
    return np.nan


def majority(s):
    s = s.dropna()
    if not len(s):
        return np.nan
    m = s.mean()
    return np.nan if m == 0.5 else float(m > 0.5)


def consensus(res):
    res = res.copy()
    res['real'] = res['is_real_glacial_lake'].map(yn) if 'is_real_glacial_lake' in res else np.nan
    res['outburst'] = res['outburst_evidence'].map(yn) if 'outburst_evidence' in res else np.nan
    g = res.groupby('lake_key')
    out = pd.DataFrame({'real': g['real'].apply(majority),
                        'outburst': g['outburst'].apply(majority),
                        'n_reviews': g['real'].apply(lambda s: int(s.notna().sum()))})
    ties = int((g['real'].mean() == 0.5).sum())
    first = g.first()
    keep = [c for c in ('feature_type', 'note') if c in first.columns]
    return out.join(first[keep]), ties


def frame_from(sheet, results):
    base = sheet if sheet is not None else results.drop_duplicates('lake_key')
    need = ['lake_key', 'area_name', 'model_score', 'in_watchlist_top2pct', 'known_glof_source',
            'design_stratum', 'inclusion_prob', 'in_design_sample']
    missing = [c for c in need if c not in base.columns]
    if missing:
        sys.exit(f'[ERROR] design columns missing from the frame: {missing}. '
                 'Regenerate the sheet with v1 and export again from v2 or the webapp.')
    f = base[need + [c for c in ('area_m2', 'area_ha') if c in base.columns]].copy()
    if 'area_ha' not in f.columns and 'area_m2' in f.columns:
        f['area_ha'] = pd.to_numeric(f['area_m2'], errors='coerce') / 1e4
    for c in ('in_watchlist_top2pct', 'known_glof_source', 'in_design_sample'):
        f[c] = pd.to_numeric(f[c].replace({True: 1, False: 0, 'True': 1, 'False': 0, 'true': 1,
                                           'false': 0}), errors='coerce').fillna(0).astype(int)
    for c in ('design_stratum', 'inclusion_prob', 'model_score'):
        f[c] = pd.to_numeric(f[c], errors='coerce')
    return f.drop_duplicates('lake_key').reset_index(drop=True)


def assign_cells(f):
    pos = f['known_glof_source'] == 1
    neg = ~pos
    wl = f['in_watchlist_top2pct'] == 1
    dsg = f['in_design_sample'] == 1
    h = f['design_stratum']
    if (neg & h.isna()).any():
        sys.exit('[ERROR] negatives without design_stratum; the sheet predates the design sample')
    hs = h.fillna(0).astype(int).map(lambda v: f'{v:02d}')
    cell = pd.Series(None, index=f.index, dtype=object)
    cell[pos] = 'pos'
    cell[neg & wl] = 'wl_d' + hs[neg & wl]
    cell[neg & ~wl & dsg] = 'd' + hs[neg & ~wl & dsg]
    f['cell'] = cell
    sizes = {'pos': int(pos.sum())}
    for c, g in f[neg & wl].groupby('cell'):
        sizes[c] = len(g)
    for d, g in f[neg & dsg].groupby('design_stratum'):
        pi = g['inclusion_prob'].dropna().unique()
        if len(pi) != 1 or not (0 < pi[0] <= 1):
            sys.exit(f'[ERROR] decile {d}: inclusion_prob must be one value in (0, 1], got {pi}')
        n_h = int(round(len(g) / pi[0]))
        w_h = int((neg & wl & (h == d)).sum())
        sizes[f'd{int(d):02d}'] = max(n_h - w_h, int((~wl.loc[g.index]).sum()))
    f['decile'] = np.where(neg, h, np.nan)
    return f, pd.Series(sizes, dtype=float)


def ratio_estimate(frame, sizes, y, x, valid):
    s = frame[frame['cell'].notna() & valid].copy()
    s['_y'] = y[s.index].astype(float)
    s['_x'] = x[s.index].astype(float)
    n_c = s.groupby('cell').size()
    w = s['cell'].map(sizes / n_c)
    X = float((w * s['_x']).sum())
    Y = float((w * s['_y'] * s['_x']).sum())
    in_dom = frame['cell'].notna() & x.astype(bool)
    dom_cells = set(frame.loc[in_dom, 'cell'])
    unsampled = sorted(c for c in dom_cells if c not in n_c.index)
    single = sorted(c for c in dom_cells if c in n_c.index and n_c[c] < 2 and n_c[c] < sizes[c])
    out = {'n': int(s.loc[s['_x'] == 1].shape[0]),
           'events_in_sample': int((s['_y'] * s['_x']).sum()), 'N_hat': X, 'total_hat': Y,
           'cells_unsampled': unsampled, 'cells_single_unit': single}
    if X <= 0:
        out.update({'estimate': np.nan, 'se': np.nan, 'ci_lo': np.nan, 'ci_hi': np.nan,
                    'se_total': np.nan})
        return out
    R = Y / X
    z = (s['_y'] * s['_x'] - R * s['_x']) / X
    t = s['_y'] * s['_x']
    v = vt = 0.0
    for c, idx in s.groupby('cell').groups.items():
        n, N = len(idx), sizes[c]
        if n < 2:
            continue
        fpc = max(0.0, 1 - n / N)
        v += N * N * fpc * z[idx].var(ddof=1) / n
        vt += N * N * fpc * t[idx].var(ddof=1) / n
    se = float(np.sqrt(v))
    out.update({'estimate': R, 'se': se, 'ci_lo': max(0.0, R - Z * se),
                'ci_hi': min(1.0, R + Z * se), 'se_total': float(np.sqrt(vt))})
    return out


def design_validation(frame, sizes, oof_path):
    real_ok = frame['real'].notna()
    commission = (frame['real'] == 0).astype(float)
    neg = frame['known_glof_source'] == 0
    out_ok = frame['outburst'].notna() | (frame['real'] == 0)
    hidden = ((frame['outburst'] == 1) & (frame['real'] != 0)).astype(float)
    ones = pd.Series(1.0, index=frame.index)
    rows = []

    def add(estimand, domain, r, **extra):
        rows.append({'estimand': estimand, 'domain': domain, **extra, **r})

    add('commission_rate', 'overall', ratio_estimate(frame, sizes, commission, ones, real_ok))
    add('commission_rate', 'known_glof_sources', ratio_estimate(
        frame, sizes, commission, (~neg).astype(float), real_ok))
    add('commission_rate', 'negatives', ratio_estimate(frame, sizes, commission, neg.astype(float), real_ok))
    add('commission_rate', 'watchlist_top2pct', ratio_estimate(
        frame, sizes, commission, (frame['in_watchlist_top2pct'] == 1).astype(float), real_ok))
    for d in sorted(frame['decile'].dropna().unique()):
        add('commission_rate', f'decile_{int(d):02d}', ratio_estimate(
            frame, sizes, commission, (frame['decile'] == d).astype(float), real_ok))

    add('missing_positive_rate', 'negatives', ratio_estimate(
        frame, sizes, hidden, neg.astype(float), out_ok))
    for d in sorted(frame['decile'].dropna().unique()):
        add('missing_positive_rate', f'decile_{int(d):02d}', ratio_estimate(
            frame, sizes, hidden, (frame['decile'] == d).astype(float), out_ok))

    pop_scores = pop_glof = pop_order = None
    if oof_path is not None and Path(oof_path).exists():
        o = pd.read_csv(oof_path)
        if 'oof_score' in o.columns:
            o = o[o['oof_score'].notna()]
            pop_scores = o['oof_score'].values
            pop_order = np.argsort(-pop_scores, kind='stable')
            if 'glof' in o.columns:
                pop_glof = o['glof'].fillna(0).astype(int).values
    if pop_scores is not None:
        N = len(pop_scores)
        n_pos_pop = int(o['glof'].sum()) if 'glof' in o.columns else None
    else:
        dsg_ok = frame['cell'].notna()
        n_c = frame[dsg_ok].groupby('cell').size()
        N = float((frame.loc[dsg_ok, 'cell'].map(sizes / n_c)).sum())
        n_pos_pop = int(sizes.get('pos', 0))
    positive = ((frame['known_glof_source'] == 1) | (hidden == 1)).astype(float)
    pos_ok = (frame['known_glof_source'] == 1) | out_ok
    known = (frame['known_glof_source'] == 1).astype(float)
    every = frame['cell'].notna()
    for pct in TOPK_PCT:
        k = max(1, int(round(N * pct / 100)))
        if pop_scores is not None:
            thr = float(np.sort(pop_scores)[::-1][k - 1])
        else:
            s = frame[every].sort_values('model_score', ascending=False)
            nc = s.groupby('cell').size()
            cw = s['cell'].map(sizes / nc).cumsum().values
            thr = float(s['model_score'].iloc[min(int(np.searchsorted(cw, k)), len(s) - 1)])
        x = (frame['model_score'] >= thr).astype(float)
        extra = {'k': k, 'top_pct': pct, 'score_threshold': thr}
        if pop_glof is not None:
            kk = int((pop_glof[pop_order][:k] == 1).sum())
            exact = {'n': k, 'events_in_sample': kk, 'N_hat': float(k), 'total_hat': float(kk),
                     'cells_unsampled': [],
                     'cells_single_unit': [], 'estimate': kk / k, 'se': 0.0, 'ci_lo': kk / k,
                     'ci_hi': kk / k, 'se_total': 0.0}
            add('precision_at_k_known_only', f'top_{pct:g}pct', exact, **extra)
            h = ratio_estimate(frame, sizes, hidden, x * neg.astype(float), out_ok)
            est = (kk + h['total_hat']) / k
            se = h['se_total'] / k
            add('precision_at_k_corrected', f'top_{pct:g}pct', {
                **h, 'events_in_sample': kk + h['events_in_sample'], 'N_hat': float(k),
                'total_hat': kk + h['total_hat'],
                'estimate': est, 'se': se, 'ci_lo': max(0.0, est - Z * se),
                'ci_hi': min(1.0, est + Z * se)}, **extra)
        else:
            add('precision_at_k_known_only', f'top_{pct:g}pct', ratio_estimate(frame, sizes, known, x, every), **extra)
            add('precision_at_k_corrected', f'top_{pct:g}pct', ratio_estimate(frame, sizes, positive, x, pos_ok), **extra)
        add('real_lake_share_at_k', f'top_{pct:g}pct', ratio_estimate(
            frame, sizes, 1 - commission, x, real_ok), **extra)
    return rows, {'population_N': N, 'known_positives': n_pos_pop,
                  'threshold_source': 'oof_scores.csv' if pop_scores is not None else 'design-weighted frame'}


def naive_summary(df, out):
    df = df[df['real'].notna()].copy()
    if not len(df):
        print('No reviewed rows (is_real_glacial_lake empty).')
        return
    df['commission'] = (df['real'] == 0).astype(int)

    def block(sub, label):
        n = len(sub)
        k = int(sub['commission'].sum())
        lo, hi = wilson(k, n)
        print(f'{label:28s} n={n:4d}  commission={k:3d}  '
              f'rate={100 * k / n:5.1f}%  95% CI [{100 * lo:.1f}, {100 * hi:.1f}]')
        return {'group': label, 'n': n, 'commission': k, 'rate_pct': round(100 * k / n, 1),
                'ci_lo': round(100 * lo, 1), 'ci_hi': round(100 * hi, 1)}

    rows = []
    print('=== COMMISSION, unweighted (all reviewed lakes, not design-based) ===')
    rows.append(block(df, 'Overall'))
    rows.append(block(df[df.in_watchlist_top2pct == 1], 'Watch-list (top-2%)'))
    if (df.in_watchlist_top2pct == 0).any():
        rows.append(block(df[df.in_watchlist_top2pct == 0], 'Non-watch-list'))
    print('--- by region ---')
    for area, sub in df.groupby('area_name'):
        rows.append(block(sub, area))
    if 'area_ha' in df.columns:
        print('--- by size ---')
        df['size_bin'] = pd.cut(pd.to_numeric(df['area_ha'], errors='coerce'), [0, 1, 5, 20, 1e9],
                                labels=['<1ha', '1-5ha', '5-20ha', '>20ha'])
        for sb, sub in df.groupby('size_bin', observed=True):
            rows.append(block(sub, f'size {sb}'))
    if 'feature_type' in df.columns:
        print('--- commission feature types ---')
        ft = df[df.commission == 1]['feature_type'].value_counts()
        print(ft.to_string() if len(ft) else '(none)')
    pd.DataFrame([r for r in rows if r['n'] > 0]).to_csv(out / 'inventory_validation_summary.csv', index=False)
    print(f'wrote {out / "inventory_validation_summary.csv"}')


def clean(o):
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    return o


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('results', nargs='?', default=str(DEFAULT_IN))
    ap.add_argument('--sheet', default=str(DEFAULT_SHEET))
    ap.add_argument('--oof', default=str(DEFAULT_OOF))
    ap.add_argument('--out-dir', default=str(DEFAULT_OUT))
    a = ap.parse_args(argv)

    path = Path(a.results)
    if not path.exists():
        print(f'No results file at {path}. Export the CSV from the review app first.')
        return
    res = pd.read_csv(path)
    sheet = pd.read_csv(a.sheet) if Path(a.sheet).exists() else None
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lab, ties = consensus(res)
    frame = frame_from(sheet, res)
    frame = frame.merge(lab, left_on='lake_key', right_index=True, how='left')
    print(f'frame: {len(frame)} lakes ({"validation sheet" if sheet is not None else "results file"}); '
          f'reviewed is_real: {int(frame["real"].notna().sum())}; '
          f'reviewed outburst flag: {int(frame["outburst"].notna().sum())}; '
          f'reviewer ties dropped: {ties}')
    naive_summary(frame, out)

    frame, sizes = assign_cells(frame)
    rows, meta = design_validation(frame, sizes, a.oof)
    print('\n=== DESIGN-BASED ESTIMATES (stratified Horvitz-Thompson, linearised SE) ===')
    for r in rows:
        k = f' k={r["k"]}' if 'k' in r else ''
        warn = ''
        if r['events_in_sample'] == 0 and r['n'] > 0:
            warn += f'  [no events in sample; one-sided 95% upper bound ~ {3 / r["n"]:.4f} (rule of three)]'
        if r['cells_unsampled']:
            warn += f'  [unsampled cells: {",".join(r["cells_unsampled"])}]'
        print(f'{r["estimand"]:27s} {r["domain"]:20s}{k:8s} n={r["n"]:4d} '
              f'N_hat={r["N_hat"]:9.1f}  est={r["estimate"]:.4f}  se={r["se"]:.4f}  '
              f'95% CI [{r["ci_lo"]:.4f}, {r["ci_hi"]:.4f}]{warn}')
    cells = pd.DataFrame({'N': sizes, 'n_reviewed_real': frame[frame['real'].notna()]
                          .groupby('cell').size()}).fillna(0)
    payload = {
        'method': ('Stratified estimator with cells = known GLOF sources (census), watch-list '
                   'negatives per score decile (census) and remaining negatives per score decile '
                   '(simple random sample from v1). Weights N_c/n_c equal 1/inclusion_prob when every '
                   'sampled lake is reviewed (Horvitz-Thompson); otherwise they treat non-reviewed '
                   'units as missing at random within the cell. Ratios use Taylor linearisation; '
                   'SE includes the finite-population correction.'),
        'inputs': {'results': str(path), 'sheet': a.sheet if sheet is not None else None,
                   'oof': a.oof if Path(a.oof).exists() else None},
        **meta,
        'cells': {k: {'N': int(v['N']), 'n_reviewed': int(v['n_reviewed_real'])} for k, v in cells.iterrows()},
        'estimates': rows,
    }
    jf = out / 'design_validation.json'
    jf.write_text(json.dumps(clean(payload), indent=2))
    flat = pd.DataFrame(rows)
    flat['cells_unsampled'] = flat['cells_unsampled'].map(lambda v: ';'.join(v))
    flat['cells_single_unit'] = flat['cells_single_unit'].map(lambda v: ';'.join(v))
    flat.round(6).to_csv(out / 'design_validation.csv', index=False)
    print(f'\nwrote {jf}')
    print(f'wrote {out / "design_validation.csv"}')


if __name__ == '__main__':
    main()
