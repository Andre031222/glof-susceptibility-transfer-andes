import sys
import json
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

LABEL_DIR = PROJECT_ROOT / 'data' / 'processed' / 'labeled'
LAKE_TABLE = LABEL_DIR / 'lake_table.csv'
ALL_EVENTS_TABLE = LABEL_DIR / 'lake_table_all_events.csv'
FORWARD_TABLE = LABEL_DIR / 'lake_table_forward.csv'
FORWARD_TRAIN_TABLE = LABEL_DIR / 'lake_table_forward_train.csv'
MODELS_DIR = PROJECT_ROOT / 'models'
FEATURE_FILE = MODELS_DIR / 'feature_sets.json'
KNNDM_FOLDS = MODELS_DIR / 'folds_knndm.csv'

PRIMARY_MODEL = 'BalancedRF'
PRIMARY_SET = 'full'
RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 50
N_REPEATS_COMPARE = 20
N_REPEATS_RANDOM = 10
MIN_REGION_POS_REPORT = 3
TOP_FRACTION = 0.10

COHORTS = {
    'primary': {'feature_set': 'full', 'filter': 'dem_before_event', 'window': 'dem'},
    'all_events_no_dem': {'feature_set': 'no_dem', 'filter': None, 'window': 'all'},
    'unambiguous': {'feature_set': 'full', 'filter': 'unambiguous', 'window': 'dem'},
    'static_only': {'feature_set': 'static', 'filter': 'dem_before_event', 'window': 'dem'},
    'bias_adjusted': {'feature_set': 'full', 'filter': 'dem_before_event', 'bias': True,
                      'window': 'dem'},
}
BIAS_FIX_QUANTILE = 0.05


def build_models(n_jobs=1):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from imblearn.ensemble import BalancedRandomForestClassifier, EasyEnsembleClassifier
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    return {
        'BalancedRF': BalancedRandomForestClassifier(
            n_estimators=500, max_features='sqrt', sampling_strategy='all', replacement=True,
            bootstrap=False, random_state=RANDOM_STATE, n_jobs=n_jobs),
        'EasyEnsemble': EasyEnsembleClassifier(n_estimators=100, random_state=RANDOM_STATE,
                                               n_jobs=n_jobs),
        'RandomForest': RandomForestClassifier(
            n_estimators=500, max_features='sqrt', class_weight='balanced',
            random_state=RANDOM_STATE, n_jobs=n_jobs),
        'XGBoost': XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                 random_state=RANDOM_STATE, eval_metric='logloss',
                                 verbosity=0, n_jobs=n_jobs),
        'LightGBM': LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                   class_weight='balanced', random_state=RANDOM_STATE,
                                   verbose=-1, n_jobs=n_jobs),
        'LogisticRegression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.1, class_weight='balanced', max_iter=2000,
                                       random_state=RANDOM_STATE))]),
    }


def make_pipeline(model, y_train):
    from sklearn.base import clone
    from sklearn.impute import SimpleImputer
    from imblearn.pipeline import Pipeline
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    steps = [('imputer', SimpleImputer(strategy='median', keep_empty_features=True))]
    if type(model).__name__ in ('XGBClassifier', 'LGBMClassifier', 'RandomForestClassifier'):
        k = max(1, min(5, int(y_train.sum()) - 1))
        steps.append(('resample', SMOTETomek(smote=SMOTE(k_neighbors=k,
                                                         random_state=RANDOM_STATE),
                                             random_state=RANDOM_STATE)))
    steps.append(('model', clone(model)))
    return Pipeline(steps)


def fit_predict(model, X_tr, y_tr, X_te, fix_cols=()):
    if len(fix_cols):
        X_te = X_te.copy()
        for c in fix_cols:
            X_te[:, c] = np.nanquantile(X_tr[:, c], BIAS_FIX_QUANTILE)
    return make_pipeline(model, y_tr).fit(X_tr, y_tr).predict_proba(X_te)[:, 1]


def metrics(y, s):
    from sklearn.metrics import roc_auc_score, average_precision_score
    if y.sum() == 0 or y.sum() == len(y):
        return {'auc': np.nan, 'pr_auc': np.nan, 'recall_top10': np.nan}
    top = np.argsort(s)[::-1][:max(1, int(np.ceil(TOP_FRACTION * len(s))))]
    return {'auc': float(roc_auc_score(y, s)), 'pr_auc': float(average_precision_score(y, s)),
            'recall_top10': float(y[top].sum() / y.sum())}


def splits(scheme, lakes, y, rep, knndm):
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    seed = RANDOM_STATE + rep
    idx = np.arange(len(y))
    if scheme == 'random':
        return list(StratifiedKFold(N_SPLITS, shuffle=True, random_state=seed).split(idx, y))
    if scheme == 'group':
        g = lakes['group'].values
        return list(StratifiedGroupKFold(N_SPLITS, shuffle=True, random_state=seed)
                    .split(idx, y, g))
    if scheme == 'knndm':
        f = knndm[knndm['rep'] == rep].set_index('lake_phys')['fold']
        fold = lakes['lake_phys'].map(f).values
        return [(idx[fold != k], idx[fold == k]) for k in np.unique(fold[~pd.isna(fold)])]
    if scheme == 'region':
        r = lakes['analysis_region'].values
        return [(idx[r != g], idx[r == g]) for g in np.unique(r)]
    raise ValueError(scheme)


def run_fold(model, X, y, tr, te, fix_cols=()):
    if y[tr].sum() < 2:
        return te, np.full(len(te), np.nan)
    return te, fit_predict(model, X[tr], y[tr], X[te], fix_cols)


def run_scheme(model, X, y, lakes, scheme, reps, knndm, n_jobs, fix_cols=()):
    tasks = [(rep, k, tr, te) for rep in range(reps)
             for k, (tr, te) in enumerate(splits(scheme, lakes, y, rep, knndm))]
    out = Parallel(n_jobs=n_jobs)(delayed(run_fold)(model, X, y, tr, te, fix_cols)
                                  for _, _, tr, te in tasks)
    oof = np.full((reps, len(y)), np.nan)
    folds = []
    for (rep, k, tr, te), (_, s) in zip(tasks, out):
        oof[rep, te] = s
        m = metrics(y[te], s) if np.isfinite(s).all() else {'auc': np.nan}
        folds.append({'rep': rep, 'fold': k, 'n_test': len(te), 'n_train': len(tr),
                      'n_pos_test': int(y[te].sum()), 'n_pos_train': int(y[tr].sum()),
                      'skipped': int(y[tr].sum() < 2), 'auc': m['auc']})
    runs = []
    for rep in range(reps):
        ok = np.isfinite(oof[rep])
        row = {'rep': rep, 'n': int(ok.sum()), 'n_pos': int(y[ok].sum()),
               'n_lost': int((~ok).sum()), 'n_pos_lost': int(y[~ok].sum()),
               'prevalence': float(y[ok].mean()), **metrics(y[ok], oof[rep, ok])}
        if scheme == 'region':
            row.update(region_summary(y, oof[rep], lakes['analysis_region'].values))
        runs.append(row)
    return oof, pd.DataFrame(runs), pd.DataFrame(folds)


def region_summary(y, s, reg):
    per = []
    for g in np.unique(reg):
        sel = (reg == g) & np.isfinite(s)
        if y[sel].sum() >= MIN_REGION_POS_REPORT:
            per.append(metrics(y[sel], s[sel]))
    per = pd.DataFrame(per, columns=['auc', 'pr_auc', 'recall_top10'])
    ok = np.isfinite(s)
    pooled = metrics(y[ok], s[ok])
    return {'auc': per['auc'].mean(), 'pr_auc': per['pr_auc'].mean(),
            'recall_top10': per['recall_top10'].mean(), 'auc_region_median': per['auc'].median(),
            'n_regions_scored': len(per), 'auc_pooled': pooled['auc'],
            'pr_auc_pooled': pooled['pr_auc']}


def report_lost(folds, runs):
    skipped = folds[folds['skipped'] == 1]
    if not len(skipped) and not runs['n_lost'].any():
        return ''
    per_rep = ', '.join(f'{int(r.rep)}:{int(r.n_lost)}' for r in runs.itertuples() if r.n_lost)
    return (f'\n      [lost] {len(skipped)} of {len(folds)} folds had < 2 training positives; '
            f'unscored lakes per rep {{{per_rep}}} (positives lost: '
            f'{int(runs["n_pos_lost"].sum())})')


def nadeau_bengio(fa, fb):
    d = (fa.set_index(['rep', 'fold'])['auc'] - fb.set_index(['rep', 'fold'])['auc']).dropna()
    n_tr = fa['n_train'].mean()
    n_te = fa['n_test'].mean()
    j = len(d)
    var = d.var(ddof=1)
    se = np.sqrt((1 / j + n_te / n_tr) * var)
    t = d.mean() / se if se > 0 else np.nan
    p = float(2 * stats.t.sf(abs(t), df=j - 1)) if np.isfinite(t) else np.nan
    return {'mean_diff': float(d.mean()), 'corrected_se': float(se), 't': float(t),
            'df': j - 1, 'p_value': p, 'n_folds': j}


def summarise(runs):
    keys = ['cohort', 'feature_set', 'model', 'scheme']
    g = runs.groupby(keys)
    out = g[['auc', 'pr_auc', 'recall_top10', 'prevalence']].mean().add_suffix('_mean')
    for q, name in ((0.025, 'p025'), (0.975, 'p975')):
        out = out.join(g[['auc', 'pr_auc']].quantile(q).add_suffix(f'_{name}'))
    out = out.join(g['rep'].count().rename('n_repeats')).join(g['n_pos'].max().rename('n_pos'))
    out = out.join(g[['n_lost', 'n_pos_lost']].mean().add_suffix('_mean'))
    extra = [c for c in ('auc_region_median', 'auc_pooled', 'pr_auc_pooled', 'n_regions_scored')
             if c in runs.columns]
    if extra:
        out = out.join(g[extra].mean())
    return out.reset_index()


def cohort_table(lakes, spec):
    if spec['filter'] == 'dem_before_event':
        return lakes[lakes['dem_before_event']].reset_index(drop=True)
    if spec['filter'] == 'unambiguous':
        keep = lakes['dem_before_event'] & ~lakes['ambiguous_match']
        return lakes[keep].reset_index(drop=True)
    return lakes.reset_index(drop=True)


def forward_validation(model, train, fwd, features, n_jobs):
    from sklearn.model_selection import StratifiedGroupKFold
    test = fwd.reset_index(drop=True)
    y_te = test['label'].values.astype(int)
    if y_te.sum() < 1 or train['label'].sum() < 3:
        return None, None
    n_splits = int(min(N_SPLITS, y_te.sum(), test['group'].nunique()))
    if n_splits < 2:
        return None, None
    split = StratifiedGroupKFold(n_splits, shuffle=True, random_state=RANDOM_STATE)
    tasks = []
    for k, (_, te) in enumerate(split.split(test, y_te, test['group'].values)):
        held = set(test['group'].iloc[te])
        tr = train[~train['group'].isin(held)]
        tasks.append((k, te, tr))
    out = Parallel(n_jobs=n_jobs)(
        delayed(run_fold)(model, np.vstack([tr[features].values.astype(float),
                                            test[features].values.astype(float)[te]]),
                          np.r_[tr['label'].values.astype(int), y_te[te]],
                          np.arange(len(tr)), np.arange(len(tr), len(tr) + len(te)))
        for _, te, tr in tasks)
    s = np.full(len(test), np.nan)
    n_train, n_train_pos, lost = [], [], 0
    for (k, te, tr), (_, sc) in zip(tasks, out):
        s[te] = sc
        n_train.append(len(tr))
        n_train_pos.append(int(tr['label'].sum()))
        if not np.isfinite(sc).all():
            lost += len(te)
    ok = np.isfinite(s)
    row = {'n_folds': len(tasks), 'n_train': float(np.mean(n_train)),
           'n_train_pos': float(np.mean(n_train_pos)), 'n_test': int(ok.sum()),
           'n_test_pos': int(y_te[ok].sum()), 'n_lost': lost, **metrics(y_te[ok], s[ok])}
    return row, pd.DataFrame({'lake_phys': test['lake_phys'].values, 'label': y_te,
                              'score': s, 'fold_group': test['group'].values})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repeats', type=int, default=N_REPEATS)
    parser.add_argument('--jobs', type=int, default=-1)
    args = parser.parse_args()

    cfg = json.loads(FEATURE_FILE.read_text())
    sets = cfg['feature_sets']
    lakes_all = pd.read_csv(LAKE_TABLE, low_memory=False)
    tables = {'dem': lakes_all, 'all': pd.read_csv(ALL_EVENTS_TABLE, low_memory=False)}
    fwd = pd.read_csv(FORWARD_TABLE, low_memory=False)
    fwd_train = pd.read_csv(FORWARD_TRAIN_TABLE, low_memory=False)
    knndm = pd.read_csv(KNNDM_FOLDS) if KNNDM_FOLDS.exists() else None
    if knndm is None:
        print(f'  [note] {KNNDM_FOLDS.name} not found: run 09_cv_folds.R for the kNNDM scheme')
    models = build_models(n_jobs=1)

    runs, folds, regions, fwd_rows = [], {}, [], []
    bias_oof = None
    primary_oof = {}
    for cohort, spec in COHORTS.items():
        lakes = cohort_table(tables[spec['window']], spec)
        fs = spec['feature_set']
        cols = list(sets[fs])
        fix_cols = ()
        if spec.get('bias'):
            bias = cfg.get('bias_features', [])
            if not bias:
                print(f'\n  cohort {cohort}: skipped (no bias covariates in the lake table)')
                continue
            fix_cols = tuple(range(len(cols), len(cols) + len(bias)))
            cols = cols + bias
        X = lakes[cols].values.astype(float)
        y = lakes['label'].values.astype(int)
        print(f'\n  cohort {cohort}: lakes {len(y)} positives {y.sum()} features {fs} '
              f'({len(cols)}{", bias covariates fixed at prediction" if fix_cols else ""})')
        names = list(models) if cohort == 'primary' else [PRIMARY_MODEL]
        schemes = ['random', 'group', 'region'] + (['knndm'] if knndm is not None else [])
        for name in names:
            for scheme in schemes:
                if scheme == 'region':
                    reps = 1
                elif scheme == 'random':
                    reps = N_REPEATS_RANDOM
                elif name == PRIMARY_MODEL:
                    reps = args.repeats
                else:
                    reps = min(N_REPEATS_COMPARE, args.repeats)
                if scheme == 'knndm':
                    reps = min(reps, int(knndm['rep'].nunique()))
                    if not lakes['lake_phys'].isin(knndm['lake_phys']).all():
                        print(f'    [skip] knndm folds do not cover cohort {cohort}')
                        continue
                oof, r, f = run_scheme(models[name], X, y, lakes, scheme, reps, knndm,
                                       args.jobs, fix_cols)
                runs.append(r.assign(cohort=cohort, feature_set=fs, model=name, scheme=scheme))
                folds[(cohort, name, scheme)] = f
                pooled = (f' (mean of {int(r["n_regions_scored"].iloc[0])} regions with >= '
                          f'{MIN_REGION_POS_REPORT} positives; pooled {r["auc_pooled"].iloc[0]:.3f})'
                          if scheme == 'region' else '')
                print(f'    {name:18s} {scheme:7s} reps={reps:3d} '
                      f'AUC={np.nanmean(r["auc"]):.3f} PR={np.nanmean(r["pr_auc"]):.3f}'
                      f'{pooled}{report_lost(f, r)}')
                if cohort == 'bias_adjusted' and scheme == 'group':
                    bias_oof = pd.Series(np.nanmean(oof, axis=0), index=lakes['lake_phys'])
                if cohort == 'primary' and name == PRIMARY_MODEL:
                    primary_oof[scheme] = np.nanmean(oof, axis=0)
                    if scheme == 'region':
                        reg = lakes['analysis_region'].values
                        for g in np.unique(reg):
                            sel = reg == g
                            m = metrics(y[sel], oof[0, sel])
                            regions.append({'held_out': g, 'n_test': int(sel.sum()),
                                            'n_pos': int(y[sel].sum()),
                                            'n_train_pos': int(y[~sel].sum()),
                                            'descriptive_only': int(y[sel].sum())
                                            < MIN_REGION_POS_REPORT, **m})
        if cohort in ('primary', 'all_events_no_dem'):
            train = fwd_train[fwd_train['window'] == spec['window']]
            row, scores = forward_validation(models[PRIMARY_MODEL], train, fwd, sets[fs],
                                             args.jobs)
            if row is not None:
                fwd_rows.append({'cohort': cohort, 'feature_set': fs, **row})
                scores.to_csv(MODELS_DIR / f'forward_scores_{cohort}.csv', index=False)
                print(f'    forward {cfg["forward_year"]}: AUC={row["auc"]:.3f} '
                      f'(test positives {row["n_test_pos"]}, {row["n_folds"]} basin folds, '
                      f'train and test lakes disjoint, lakes lost {row["n_lost"]})')

    runs = pd.concat(runs, ignore_index=True)
    runs.to_csv(MODELS_DIR / 'cv_runs.csv', index=False)
    summarise(runs).to_csv(MODELS_DIR / 'cv_summary.csv', index=False)
    pd.concat([f.assign(cohort=c, model=m, scheme=s) for (c, m, s), f in folds.items()],
              ignore_index=True).to_csv(MODELS_DIR / 'cv_folds.csv', index=False)
    pd.DataFrame(regions).to_csv(MODELS_DIR / 'transfer_by_region.csv', index=False)
    pd.DataFrame(fwd_rows).to_csv(MODELS_DIR / 'forward_validation.csv', index=False)

    tests = []
    for scheme in ('group', 'knndm'):
        base = folds.get(('primary', PRIMARY_MODEL, scheme))
        if base is None:
            continue
        for name in models:
            other = folds.get(('primary', name, scheme))
            if name == PRIMARY_MODEL or other is None:
                continue
            reps = other['rep'].max() + 1
            tests.append({'scheme': scheme, 'model_a': PRIMARY_MODEL, 'model_b': name,
                          **nadeau_bengio(base[base['rep'] < reps], other)})
    pd.DataFrame(tests).to_csv(MODELS_DIR / 'model_tests.csv', index=False)

    lakes = cohort_table(lakes_all, COHORTS['primary'])
    oof = lakes[['lake_phys', 'country', 'analysis_region', 'group', 'lon', 'lat', 'cutoff',
                 'area_m2', 'label']].rename(columns={'lake_phys': 'lake_key', 'label': 'glof'})
    for scheme, s in primary_oof.items():
        oof[f'oof_{scheme}'] = np.round(s, 5)
    oof['oof_score'] = oof.get('oof_knndm', oof['oof_group'])
    if bias_oof is not None:
        oof['oof_bias_adjusted'] = np.round(oof['lake_key'].map(bias_oof).values, 5)
        rho = stats.spearmanr(oof['oof_group'], oof['oof_bias_adjusted'], nan_policy='omit')
        print(f'  rank agreement primary vs bias-adjusted (Spearman): {rho.statistic:.3f}')
    oof.to_csv(MODELS_DIR / 'oof_scores.csv', index=False)

    import joblib
    X = lakes[sets[PRIMARY_SET]].values.astype(float)
    y = lakes['label'].values.astype(int)
    final = make_pipeline(build_models(n_jobs=-1)[PRIMARY_MODEL], y).fit(X, y)
    joblib.dump({'model': final, 'feature_cols': sets[PRIMARY_SET], 'name': PRIMARY_MODEL},
                MODELS_DIR / 'primary_model.joblib')
    (MODELS_DIR / 'run_summary.json').write_text(json.dumps({
        'primary_model': PRIMARY_MODEL, 'primary_feature_set': PRIMARY_SET,
        'cohorts': COHORTS, 'feature_sets': sets, 'n_splits': N_SPLITS,
        'n_repeats': args.repeats, 'n_repeats_compare': N_REPEATS_COMPARE,
        'group_level': cfg['group_level'], 'knndm': knndm is not None,
        'n_lakes_primary': int(len(y)), 'n_pos_primary': int(y.sum())}, indent=2))
    print('\n  saved cv_runs.csv, cv_summary.csv, cv_folds.csv, model_tests.csv, '
          'transfer_by_region.csv, forward_validation.csv, oof_scores.csv, '
          'primary_model.joblib, run_summary.json')


if __name__ == '__main__':
    main()
