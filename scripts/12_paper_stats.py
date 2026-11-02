import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(os.environ.get('GLOF_ROOT') or Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

MODELS_DIR = PROJECT_ROOT / 'models'
LAKE_TABLE = PROJECT_ROOT / 'data' / 'processed' / 'labeled' / 'lake_table.csv'
FORWARD_TABLE = PROJECT_ROOT / 'data' / 'processed' / 'labeled' / 'lake_table_forward.csv'
INPUTS = {
    'oof': MODELS_DIR / 'oof_scores.csv',
    'run': MODELS_DIR / 'run_summary.json',
    'features': MODELS_DIR / 'feature_sets.json',
    'cv_summary': MODELS_DIR / 'cv_summary.csv',
    'transfer': MODELS_DIR / 'transfer_by_region.csv',
    'forward': MODELS_DIR / 'forward_validation.csv',
    'tests': MODELS_DIR / 'model_tests.csv',
}
OPTIONAL = {
    'knndm_w': MODELS_DIR / 'knndm_W.csv',
    'geodist': MODELS_DIR / 'geodist.csv',
    'aoa': MODELS_DIR / 'aoa_by_region.csv',
    'spatial': MODELS_DIR / 'spatial_summary.json',
}
OUTPUTS = {
    'stats': MODELS_DIR / 'paper_stats.json',
    'calibrated': MODELS_DIR / 'oof_calibrated.csv',
    'performance': MODELS_DIR / 'performance_by_scheme.csv',
    'watch_list': MODELS_DIR / 'watch_list.csv',
    'thresholds': MODELS_DIR / 'threshold_sensitivity.csv',
    'mann_whitney': MODELS_DIR / 'mann_whitney.csv',
    'by_region': MODELS_DIR / 'summary_by_region.csv',
    'by_country': MODELS_DIR / 'summary_by_country.csv',
}
RANDOM_STATE = 42
N_CAL_SPLITS = 5
N_BOOTSTRAP = 1000
CI = (2.5, 97.5)
THRESHOLDS = [0.01, 0.02, 0.05, 0.10, 0.20, 0.30, 0.50]
SCHEME_ORDER = ['random', 'group', 'knndm', 'region', 'forward']
MIN_REGION_POS = 3
TROPIC_LAT = -18.0
GEODIST_KEY = {'random': 'CV-random', 'group': 'CV-group', 'knndm': 'CV-knndm'}


def load_inputs():
    for f in [LAKE_TABLE, *INPUTS.values()]:
        if not f.exists():
            sys.exit(f'[ERROR] {f} not found; run 08_lake_table.py and 10_train_validate.py first')
    run = json.loads(INPUTS['run'].read_text())
    cfg = json.loads(INPUTS['features'].read_text())
    oof = pd.read_csv(INPUTS['oof'])
    lakes = pd.read_csv(LAKE_TABLE, low_memory=False).set_index('lake_phys').reindex(oof['lake_key'])
    if lakes['label'].isna().any():
        raise ValueError('oof_scores.csv lists lakes absent from lake_table.csv; re-run 10')
    if not np.array_equal(lakes['label'].values.astype(int), oof['glof'].values.astype(int)):
        raise ValueError('labels differ between oof_scores.csv and lake_table.csv; re-run 10')
    lakes = lakes.reset_index().rename(columns={'lake_key': 'lake_phys'})
    tables = {k: read_csv_safe(INPUTS[k]) for k in ('cv_summary', 'transfer', 'forward', 'tests')}
    extra = {k: (read_csv_safe(f) if f.suffix == '.csv' else json.loads(f.read_text()))
             for k, f in OPTIONAL.items() if f.exists()}
    return lakes, oof, run, cfg, tables, extra


def read_csv_safe(path):
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def youden_threshold(y, s):
    from sklearn.metrics import roc_curve
    ok = np.isfinite(s)
    fpr, tpr, thr = roc_curve(y[ok], s[ok])
    return float(thr[np.argmax(tpr - fpr)])


def cluster_bootstrap(y, s, groups, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    from sklearn.metrics import roc_auc_score, average_precision_score
    ok = np.isfinite(s)
    y, s = y[ok].astype(int), s[ok]
    codes, uniq = pd.factorize(pd.Series(groups[ok]).astype(str))
    n_groups = len(uniq)
    out = {'n_scored': int(ok.sum()), 'n_pos_scored': int(y.sum()), 'n_clusters': n_groups,
           'n_boot': 0}
    if y.sum() == 0 or y.sum() == len(y):
        return out
    out.update({'auc_oof': float(roc_auc_score(y, s)),
                'pr_auc_oof': float(average_precision_score(y, s)),
                'pr_baseline': float(y.mean())})
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)[codes]
        keep = w > 0
        yb, sb, wb = y[keep], s[keep], w[keep]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        draws.append((roc_auc_score(yb, sb, sample_weight=wb),
                      average_precision_score(yb, sb, sample_weight=wb),
                      float((wb * yb).sum() / wb.sum())))
    if draws:
        d = np.asarray(draws)
        out['n_boot'] = len(d)
        for j, k in enumerate(('auc', 'pr_auc', 'pr_baseline')):
            lo, hi = np.percentile(d[:, j], CI)
            out[f'{k}_lo'], out[f'{k}_hi'] = float(lo), float(hi)
    return out


def bootstrap_scores(oof, tables, model_name):
    rows = []
    groups = oof['group'].values
    y = oof['glof'].values.astype(int)
    for scheme in SCHEME_ORDER:
        col = f'oof_{scheme}'
        if col not in oof:
            continue
        b = cluster_bootstrap(y, oof[col].values.astype(float), groups)
        rows.append({'cohort': 'primary', 'model': model_name, 'scheme': scheme, **b})
        print(f'    bootstrap {scheme:8s} clusters={b["n_clusters"]}  reps={b["n_boot"]}')
    fwd_groups = None
    if FORWARD_TABLE.exists():
        f = pd.read_csv(FORWARD_TABLE, usecols=['lake_phys', 'group'], low_memory=False)
        fwd_groups = f.drop_duplicates('lake_phys').set_index('lake_phys')['group']
    for cohort in tables['forward'].get('cohort', pd.Series(dtype=str)).unique():
        path = MODELS_DIR / f'forward_scores_{cohort}.csv'
        if not path.exists():
            continue
        sc = pd.read_csv(path)
        g = (sc['lake_phys'].map(fwd_groups) if fwd_groups is not None
             else pd.Series(np.nan, index=sc.index))
        g = g.astype(object).where(g.notna(), 'lake_' + sc['lake_phys'].astype(str)).values
        b = cluster_bootstrap(sc['label'].values.astype(int), sc['score'].values.astype(float), g)
        rows.append({'cohort': cohort, 'model': model_name, 'scheme': 'forward', **b})
        print(f'    bootstrap forward  {cohort}  clusters={b["n_clusters"]}  reps={b["n_boot"]}')
    return pd.DataFrame(rows)


def within_region_auc(oof, score_col='oof_score', n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    from sklearn.metrics import roc_auc_score
    y = oof['glof'].values.astype(int)
    s = oof[score_col].values.astype(float)
    reg = oof['analysis_region'].values
    grp = oof['group'].astype(str).values
    keep = [r for r in pd.unique(reg) if y[reg == r].sum() >= MIN_REGION_POS
            and y[reg == r].sum() < (reg == r).sum()]
    s_out = oof['oof_region'].values.astype(float) if 'oof_region' in oof else None
    rows = []
    for r in keep:
        m = reg == r
        row = {'region': r, 'n': int(m.sum()), 'n_pos': int(y[m].sum()),
               'auc_within': float(roc_auc_score(y[m], s[m]))}
        b = cluster_bootstrap(y[m], s[m], grp[m], n_boot=n_boot, seed=seed)
        row['auc_within_lo'], row['auc_within_hi'] = b.get('auc_lo'), b.get('auc_hi')
        if s_out is not None:
            b = cluster_bootstrap(y[m], s_out[m], grp[m], n_boot=n_boot, seed=seed)
            row['auc_heldout'] = b.get('auc_oof')
            row['auc_heldout_lo'], row['auc_heldout_hi'] = b.get('auc_lo'), b.get('auc_hi')
        rows.append(row)

    def stat(yy, ss, rr, ww):
        a, n = [], []
        for r in keep:
            m = rr == r
            if 0 < yy[m].sum() < m.sum():
                a.append(roc_auc_score(yy[m], ss[m], sample_weight=ww[m]))
                n.append((ww[m] * yy[m]).sum())
        return (np.mean(a), np.average(a, weights=n)) if a else (np.nan, np.nan)

    point = stat(y, s, reg, np.ones(len(y)))
    codes, uniq = pd.factorize(pd.Series(grp))
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, len(uniq), len(uniq)), minlength=len(uniq))[codes]
        k = w > 0
        draws.append(stat(y[k], s[k], reg[k], w[k].astype(float)))
    d = np.asarray(draws, dtype=float)
    lo, hi = np.nanpercentile(d, CI, axis=0)
    out = {'min_pos': MIN_REGION_POS, 'n_regions': len(rows), 'regions': rows,
           'auc_within_mean': float(point[0]), 'auc_within_mean_lo': float(lo[0]),
           'auc_within_mean_hi': float(hi[0]), 'auc_within_posweighted': float(point[1]),
           'auc_within_posweighted_lo': float(lo[1]), 'auc_within_posweighted_hi': float(hi[1])}
    for name, m in (('tropical', oof['lat'].values > TROPIC_LAT),
                    ('extratropical', oof['lat'].values <= TROPIC_LAT)):
        b = cluster_bootstrap(y[m], s[m], grp[m], n_boot=n_boot, seed=seed)
        out[name] = {k: b.get(k) for k in ('n_scored', 'n_pos_scored', 'auc_oof', 'auc_lo',
                                           'auc_hi', 'pr_auc_oof', 'pr_baseline')}
    return out


def watch_list_table(y, proba):
    order = np.argsort(-proba)
    y_sorted = y[order]
    n = len(y)
    total_pos = int(y.sum())
    base = total_pos / n
    rows = []
    for pct in [0.5, 1.0, 2.0, 5.0, 10.0, 20.0]:
        k = max(1, int(round(n * pct / 100)))
        caught = int(y_sorted[:k].sum())
        prec = caught / k
        rows.append({'top_pct': pct, 'n_lakes': k, 'pos_caught': caught,
                     'recall': round(caught / total_pos, 3) if total_pos else 0.0,
                     'lift': round(prec / base, 2) if base else 0.0})
    return rows


def threshold_table(y, proba):
    total_pos = int(y.sum())
    n = len(y)
    rows = []
    for t in THRESHOLDS:
        flagged = proba >= t
        caught = int(y[flagged].sum())
        rows.append({'threshold': t, 'n_flagged': int(flagged.sum()),
                     'pct_flagged': round(100 * flagged.sum() / n, 1),
                     'recall': round(caught / total_pos, 3) if total_pos else 0.0,
                     'pos_caught': caught})
    return rows


def mann_whitney(df, feature_cols, y):
    pos = df[y == 1]
    neg = df[y == 0]
    out = []
    for c in feature_cols:
        a = pos[c].dropna().values
        b = neg[c].dropna().values
        if len(a) < 3 or len(b) < 3:
            continue
        u, p = stats.mannwhitneyu(a, b, alternative='two-sided')
        out.append({'feature': c, 'median_glof': float(np.median(a)),
                    'median_nonglof': float(np.median(b)), 'U': float(u), 'p_value': float(p),
                    'rank_biserial': round(float(2 * u / (len(a) * len(b)) - 1), 3),
                    'n_pos': len(a)})
    out.sort(key=lambda r: r['p_value'])
    for r in out:
        r['p_bonferroni'] = min(1.0, r['p_value'] * len(out))
        r['significant'] = r['p_bonferroni'] < 0.05
    return out


def calibrate(y, proba):
    from sklearn.metrics import brier_score_loss
    from sklearn.model_selection import StratifiedKFold
    from sklearn.isotonic import IsotonicRegression
    ok = np.isfinite(proba)
    base = y[ok].mean()
    brier = brier_score_loss(y[ok], proba[ok])
    brier_ref = brier_score_loss(y[ok], np.full(int(ok.sum()), base))
    cal = np.full(len(y), np.nan)
    idx = np.flatnonzero(ok)
    skf = StratifiedKFold(n_splits=N_CAL_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in skf.split(idx.reshape(-1, 1), y[idx]):
        ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0)
        cal[idx[te]] = ir.fit(proba[idx[tr]], y[idx[tr]]).predict(proba[idx[te]])
    cal = np.nan_to_num(cal, nan=base)
    brier_cal = brier_score_loss(y[ok], cal[ok])
    summary = {'brier': round(float(brier), 5), 'brier_ref': round(float(brier_ref), 5),
               'brier_skill_score': round(float(1 - brier / brier_ref), 4) if brier_ref else 0.0,
               'base_rate': round(float(base), 5), 'brier_calibrated': round(float(brier_cal), 5),
               'n_unscored': int((~ok).sum()),
               'brier_skill_score_calibrated': (round(float(1 - brier_cal / brier_ref), 4)
                                                if brier_ref else 0.0),
               'method': f'isotonic regression, {N_CAL_SPLITS}-fold cross-fitted on OOF scores'}
    return cal, summary


def ks_test(y, proba):
    ok = np.isfinite(proba)
    a, b = proba[ok & (y == 1)], proba[ok & (y == 0)]
    d, p = stats.ks_2samp(a, b)
    return {'ks_D': round(float(d), 4), 'p_value': float(p),
            'median_score_glof': round(float(np.median(a)), 3),
            'median_score_nonglof': round(float(np.median(b)), 3)}


def group_summary(df, key, threshold):
    from sklearn.metrics import roc_auc_score
    q90 = df['oof_score'].quantile(0.90)
    rows = []
    for g, d in df.groupby(key):
        y, s = d['glof'].values.astype(int), d['oof_score'].values.astype(float)
        ok = np.isfinite(s)
        yo = y[ok]
        rows.append({key: g, 'n_lakes': int(len(d)), 'n_pos': int(y.sum()),
                     'prevalence': round(float(y.mean()), 5),
                     'median_score': round(float(np.nanmedian(s)), 4) if ok.any() else np.nan,
                     'pct_flagged': round(100 * float((d['calibrated'].values >= threshold)
                                                      .mean()), 1),
                     'n_top_decile': int((s >= q90).sum()),
                     'pct_top_decile': round(100 * float((s >= q90).mean()), 1),
                     'auc_within': (float(roc_auc_score(yo, s[ok])) if 0 < yo.sum() < len(yo)
                                    else np.nan)})
    return pd.DataFrame(rows).sort_values('n_lakes', ascending=False)


def performance_table(cv, fwd, extra, model_name, boot, feature_set):
    rows = []
    w = extra.get('knndm_w')
    geo = extra.get('geodist')
    for _, r in cv.iterrows():
        row = {'cohort': r['cohort'], 'feature_set': r['feature_set'], 'model': r['model'],
               'scheme': r['scheme'], 'n_repeats': r.get('n_repeats', np.nan),
               'n_pos': r.get('n_pos', np.nan),
               'prevalence': r.get('prevalence_mean', np.nan), 'auc': r.get('auc_mean', np.nan),
               'auc_rep_lo': r.get('auc_p025', np.nan), 'auc_rep_hi': r.get('auc_p975', np.nan),
               'pr_auc': r.get('pr_auc_mean', np.nan),
               'pr_auc_rep_lo': r.get('pr_auc_p025', np.nan),
               'pr_auc_rep_hi': r.get('pr_auc_p975', np.nan),
               'recall_top10': r.get('recall_top10_mean', np.nan), 'W': np.nan,
               'median_nn_km': np.nan}
        if w is not None:
            if r['scheme'] == 'knndm':
                row['W'] = float(w.loc[w['method'].str.startswith('knndm'), 'W'].median())
            elif r['scheme'] in ('random', 'group'):
                row['W'] = float(w.loc[w['method'] == r['scheme'], 'W'].median())
        if geo is not None and r['scheme'] in GEODIST_KEY:
            d = geo.loc[geo['what'] == GEODIST_KEY[r['scheme']], 'distance_m']
            row['median_nn_km'] = float(d.median() / 1000) if len(d) else np.nan
        rows.append(row)
    for _, r in fwd.iterrows():
        rows.append({'cohort': r['cohort'], 'feature_set': r['feature_set'],
                     'model': model_name, 'scheme': 'forward', 'n_repeats': 1,
                     'n_pos': int(r['n_test_pos']), 'prevalence': r['n_test_pos'] / r['n_test'],
                     'auc': r['auc'], 'auc_rep_lo': np.nan, 'auc_rep_hi': np.nan,
                     'pr_auc': r['pr_auc'], 'pr_auc_rep_lo': np.nan, 'pr_auc_rep_hi': np.nan,
                     'recall_top10': r['recall_top10'], 'W': np.nan, 'median_nn_km': np.nan})
    out = pd.DataFrame(rows)
    keys = ['cohort', 'model', 'scheme']
    if not boot.empty:
        have = set(map(tuple, out[keys].values)) if not out.empty else set()
        extra_rows = [{'cohort': b['cohort'], 'feature_set': feature_set, 'model': b['model'],
                       'scheme': b['scheme'], 'n_repeats': np.nan, 'n_pos': b['n_pos_scored'],
                       'prevalence': b.get('pr_baseline', np.nan),
                       'auc': b.get('auc_oof', np.nan), 'pr_auc': b.get('pr_auc_oof', np.nan)}
                      for _, b in boot.iterrows()
                      if (b['cohort'], b['model'], b['scheme']) not in have]
        out = pd.concat([out, pd.DataFrame(extra_rows)], ignore_index=True)
        out = out.merge(boot, on=keys, how='left')
        out['ci_method'] = np.where(out['n_boot'].fillna(0) > 0,
                                    f'cluster bootstrap over sub-basins ({N_BOOTSTRAP} draws, '
                                    'pooled OOF scores)', '')
    if out.empty:
        return out
    out['scheme_rank'] = out['scheme'].map({s: i for i, s in enumerate(SCHEME_ORDER)})
    return out.sort_values(['cohort', 'model', 'scheme_rank']).drop(columns='scheme_rank')


def clean_json(o):
    if isinstance(o, dict):
        return {k: clean_json(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean_json(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer, np.bool_)):
        return o.item()
    return o


def main():
    print('=== Paper stats ===')
    lakes, oof, run, cfg, tables, extra = load_inputs()
    model_name = run['primary_model']
    feature_set = run['primary_feature_set']
    feature_cols = cfg['feature_sets'][feature_set]
    y = oof['glof'].values.astype(int)
    s = oof['oof_score'].values.astype(float)
    headline = 'knndm' if 'oof_knndm' in oof else 'group'
    print(f'  lakes={len(y)}  pos={int(y.sum())}  model={model_name} ({feature_set}, '
          f'{len(feature_cols)} features)  oof_score from the {headline} scheme')

    cal, cal_summary = calibrate(y, s)
    pd.DataFrame({'lake_key': oof['lake_key'], 'oof_score': s,
                  'calibrated': np.round(cal, 6)}).to_csv(OUTPUTS['calibrated'], index=False)
    t = youden_threshold(y, cal)
    t_raw = youden_threshold(y, s)

    boot = bootstrap_scores(oof, tables, model_name)
    perf = performance_table(tables['cv_summary'], tables['forward'], extra, model_name, boot,
                             feature_set)
    perf.to_csv(OUTPUTS['performance'], index=False)
    prim = perf[(perf['cohort'] == 'primary') & (perf['model'] == model_name)]

    tab = oof[['lake_key', 'glof', 'oof_score', 'analysis_region', 'country']].assign(
        calibrated=cal)
    by_region = group_summary(tab, 'analysis_region', t)
    by_country = group_summary(tab, 'country', t)
    by_region.to_csv(OUTPUTS['by_region'], index=False)
    by_country.to_csv(OUTPUTS['by_country'], index=False)

    transfer = tables['transfer']
    if 'aoa' in extra and not extra['aoa'].empty and not transfer.empty:
        transfer = transfer.merge(extra['aoa'][['held_out', 'frac_inside_aoa', 'di_median',
                                                'auc_inside_aoa', 'auc_outside_aoa']],
                                  on='held_out', how='left')
    sens = perf[perf['model'] == model_name]

    results = {
        'model': model_name, 'feature_set': feature_set, 'headline_scheme': headline,
        'n_lakes': int(len(y)), 'n_pos': int(y.sum()), 'n_neg': int((y == 0).sum()),
        'n_repeats': run.get('n_repeats'), 'group_level': run.get('group_level'),
        'performance_primary': prim.to_dict(orient='records'),
        'headline': (prim[prim['scheme'] == headline].iloc[0].to_dict()
                     if (prim['scheme'] == headline).any() else None),
        'transfer_by_region': transfer.to_dict(orient='records'),
        'forward_validation': tables['forward'].to_dict(orient='records'),
        'model_tests': tables['tests'].to_dict(orient='records'),
        'sensitivity_cohorts': sens.to_dict(orient='records'),
        'youden_threshold': round(t, 4),
        'youden_threshold_scale': 'calibrated probability (oof_calibrated.csv)',
        'youden_threshold_raw_score': round(t_raw, 4),
        'watch_list': watch_list_table(y, s),
        'threshold_sensitivity': threshold_table(y, cal),
        'calibration': cal_summary,
        'ks_test': ks_test(y, s),
        'mann_whitney': mann_whitney(lakes, feature_cols, y),
        'pct_high_risk_overall': round(100 * float((cal >= t).mean()), 1),
        'by_region': by_region.to_dict(orient='records'),
        'by_country': by_country.to_dict(orient='records'),
        'within_region': within_region_auc(oof),
    }
    if 'spatial' in extra:
        sp = extra['spatial']
        results['spatial'] = {k: sp.get(k) for k in ('residual_range_km',
                                                     'predictor_range_km_median',
                                                     'aoa_pooled_fraction_inside')}

    OUTPUTS['stats'].write_text(json.dumps(clean_json(results), indent=2, allow_nan=False))
    pd.DataFrame(results['watch_list']).to_csv(OUTPUTS['watch_list'], index=False)
    pd.DataFrame(results['threshold_sensitivity']).to_csv(OUTPUTS['thresholds'], index=False)
    pd.DataFrame(results['mann_whitney']).to_csv(OUTPUTS['mann_whitney'], index=False)

    print('\n  Primary model by scheme (mean over repetitions; pooled OOF with 95% cluster '
          'bootstrap CI):')
    for _, r in prim.iterrows():
        print(f'    {r["scheme"]:8s} AUC={r["auc"]:.3f}  pooled={r.get("auc_oof", np.nan):.3f} '
              f'[{r.get("auc_lo", np.nan):.3f}, {r.get("auc_hi", np.nan):.3f}]  '
              f'PR-AUC={r.get("pr_auc_oof", np.nan):.3f} '
              f'[{r.get("pr_auc_lo", np.nan):.3f}, {r.get("pr_auc_hi", np.nan):.3f}]  '
              f'baseline={r.get("pr_baseline", np.nan):.4f}  n_pos={r["n_pos"]}  '
              f'reps={r["n_repeats"]}')
    wr = results['within_region']
    print(f'  within-region AUC (mean of {wr["n_regions"]} regions with >= {MIN_REGION_POS} '
          f'positives) = {wr["auc_within_mean"]:.3f} [{wr["auc_within_mean_lo"]:.3f}, '
          f'{wr["auc_within_mean_hi"]:.3f}]; positive-weighted {wr["auc_within_posweighted"]:.3f}')
    for k in ('tropical', 'extratropical'):
        b = wr[k]
        print(f'  {k:13s} lakes={b["n_scored"]} pos={b["n_pos_scored"]} AUC={b.get("auc_oof") or float("nan"):.3f} '
              f'[{b.get("auc_lo") or float("nan"):.3f}, {b.get("auc_hi") or float("nan"):.3f}]')
    print(f'\n  Youden threshold (calibrated)={t:.3f} (raw score {t_raw:.3f})  '
          f'high-risk overall={results["pct_high_risk_overall"]}%')
    for r in results['watch_list']:
        print(f'    top {r["top_pct"]:4}% : {r["n_lakes"]:5} lakes, recall {r["recall"]}, '
              f'lift {r["lift"]}x')
    c = results['calibration']
    print(f'  Calibration BSS raw={c["brier_skill_score"]}  '
          f'calibrated={c["brier_skill_score_calibrated"]}  '
          f'KS D={results["ks_test"]["ks_D"]} p={results["ks_test"]["p_value"]:.4f}')
    if not tables['tests'].empty:
        print('  Nadeau-Bengio corrected t-tests (primary minus comparator):')
        for _, r in tables['tests'].iterrows():
            print(f'    {r["scheme"]:6s} {r["model_b"]:18s} dAUC={r["mean_diff"]:+.4f} '
                  f'p={r["p_value"]:.3f}')
    print('\n  Saved: ' + ', '.join(p.name for p in OUTPUTS.values()))


if __name__ == '__main__':
    main()
