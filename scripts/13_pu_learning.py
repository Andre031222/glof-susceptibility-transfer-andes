import os
import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get('GLOF_ROOT') or Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

MODELS_DIR = PROJECT_ROOT / 'models'
LAKE_TABLE = PROJECT_ROOT / 'data' / 'processed' / 'labeled' / 'lake_table.csv'
OUT = MODELS_DIR / 'pu_results.json'
RANDOM_STATE = 42
N_SPLITS = 5
N_BOOTSTRAP = 500


def load_design(lake_keys):
    run = json.loads((MODELS_DIR / 'run_summary.json').read_text())
    cfg = json.loads((MODELS_DIR / 'feature_sets.json').read_text())
    feature_cols = cfg['feature_sets'][run['primary_feature_set']]
    lakes = pd.read_csv(LAKE_TABLE, low_memory=False).set_index('lake_phys')
    lakes = lakes.reindex(lake_keys)
    if lakes['label'].isna().any():
        raise ValueError('oof_scores.csv and lake_table.csv list different lakes; re-run 10')
    X = lakes[feature_cols].values.astype(float)
    y = lakes['label'].values.astype(int)
    groups = lakes['group'].astype(str).values
    return X, y, groups


def auc_rank(y, s):
    from sklearn.metrics import roc_auc_score
    if y.sum() == 0 or (y == 0).sum() == 0:
        return np.nan
    return float(roc_auc_score(y, s))


def unbalance(b, n_labelled, n_unlabelled):
    r = n_labelled / n_unlabelled
    return r * b / (r * b + 1.0 - b)


def heldout_label_probability(X, y, groups, estimator):
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.impute import SimpleImputer
    from sklearn.base import clone
    skf = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    held = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y, groups):
        imp = SimpleImputer(strategy='median', keep_empty_features=True)
        clf = clone(estimator).fit(imp.fit_transform(X[tr]), y[tr])
        held[te] = clf.predict_proba(imp.transform(X[te]))[:, 1]
    return held


def pi_unlabelled(pi, c):
    if not (np.isfinite(pi) and np.isfinite(c)):
        return np.nan
    return float(pi * (1 - c) / (1 - pi * c))


def elkan_noto_c(scores_pos_heldout):
    v = scores_pos_heldout[np.isfinite(scores_pos_heldout)]
    if v.size == 0:
        return np.nan
    return float(np.mean(v))


def prior_from_c(n_labelled, n_total, c):
    if not np.isfinite(c) or c <= 0:
        return np.nan
    return float(min(1.0, (n_labelled / n_total) / c))


def prior_km(scores, y, quantiles=(0.90, 0.95, 0.99)):
    out = {}
    unl = scores[y == 0]
    pos = scores[y == 1]
    for q in quantiles:
        t = np.nanquantile(unl, q)
        recall = float(np.mean(pos >= t)) if pos.size else np.nan
        frac_unl = float(np.mean(unl >= t))
        out[f'q{int(q*100)}'] = {
            'threshold': round(float(t), 4),
            'recall_pos': round(recall, 4),
            'frac_unlabelled_above': round(frac_unl, 5),
            'pi_upper': round(frac_unl / recall, 5) if recall > 0 else None,
        }
    return out


def corrected_auc(auc_pu, pi_u):
    if not np.isfinite(auc_pu) or not np.isfinite(pi_u) or pi_u >= 1:
        return np.nan
    return float(min(1.0, (auc_pu - 0.5 * pi_u) / (1 - pi_u)))


def bootstrap_c(held, y, s, groups, n_boot=N_BOOTSTRAP, seed=RANDOM_STATE):
    from sklearn.metrics import roc_auc_score
    codes, uniq = pd.factorize(pd.Series(groups).astype(str))
    n_groups = len(uniq)
    rng = np.random.default_rng(seed)
    pos = y == 1
    draws = []
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n_groups, n_groups), minlength=n_groups)[codes]
        n_lab_b = float(w[pos].sum())
        if n_lab_b == 0 or n_lab_b == w.sum():
            continue
        c_b = []
        for h in held.values():
            ok = pos & np.isfinite(h) & (w > 0)
            c_b.append(float(np.average(h[ok], weights=w[ok])) if ok.any() else np.nan)
        c_b = float(np.nanmedian(c_b))
        pi_b = prior_from_c(n_lab_b, float(w.sum()), c_b)
        pi_u_b = pi_unlabelled(pi_b, c_b)
        keep = w > 0
        auc_b = float(roc_auc_score(y[keep], s[keep], sample_weight=w[keep]))
        draws.append((c_b, pi_b, pi_u_b, corrected_auc(auc_b, pi_u_b)))
    d = np.asarray(draws, dtype=float).reshape(-1, 4)
    out = {'n_boot': int(len(d)), 'n_clusters': int(n_groups)}
    for j, k in enumerate(('c', 'pi', 'pi_unlabelled', 'auc_corrected')):
        v = d[:, j][np.isfinite(d[:, j])]
        out[k] = ([round(float(q), 6) for q in np.percentile(v, (2.5, 97.5))]
                  if v.size else None)
    return out


def operating_points(scores, y, pi, fractions=(0.005, 0.01, 0.02, 0.05, 0.10, 0.20)):
    n = len(y)
    obs_prev = float(y.mean())
    order = np.argsort(-scores)
    ys = y[order]
    rows = []
    for f in fractions:
        k = max(1, int(round(n * f)))
        caught = int(ys[:k].sum())
        prec_obs = caught / k
        recall_obs = caught / max(int(y.sum()), 1)
        rows.append({
            'top_frac': f,
            'n': k,
            'labelled_caught': caught,
            'precision_observed': round(prec_obs, 5),
            'lift_observed': round(prec_obs / obs_prev, 2) if obs_prev else None,
            'precision_pu': (round(min(1.0, prec_obs * pi / obs_prev), 5)
                             if np.isfinite(pi) and obs_prev else None),
            'recall_observed': round(recall_obs, 4),
        })
    return rows


def main():
    print('=== PU learning ===')
    oof_file = MODELS_DIR / 'oof_scores.csv'
    required = (oof_file, MODELS_DIR / 'run_summary.json', MODELS_DIR / 'feature_sets.json',
                LAKE_TABLE)
    for f in required:
        if not f.exists():
            sys.exit(f'[ERROR] {f} not found; run 08_lake_table.py and 10_train_validate.py first')

    oof = pd.read_csv(oof_file)
    y = oof['glof'].values.astype(int)
    s = oof['oof_score'].values.astype(float)
    ok = np.isfinite(s)
    y, s = y[ok], s[ok]
    n, n_pos = len(y), int(y.sum())
    obs_prev = n_pos / n

    print(f'  lakes={n}  labelled positives={n_pos}  observed prevalence={obs_prev:.5f}')

    if n_pos < 5:
        print('[ERROR] too few positives for PU estimation')
        sys.exit(1)

    X, y_full, groups = load_design(oof['lake_key'].values[ok])
    if not np.array_equal(y_full, y):
        raise ValueError('labels differ between oof_scores.csv and lake_table.csv; re-run 10')

    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from imblearn.ensemble import BalancedRandomForestClassifier

    n_lab = int(y_full.sum())
    n_unl = int((y_full == 0).sum())
    estimators = {
        'logistic_unweighted': (Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, max_iter=2000, random_state=RANDOM_STATE))]),
            False),
        'balanced_rf_prior_corrected': (BalancedRandomForestClassifier(
            n_estimators=500, max_features='sqrt', random_state=RANDOM_STATE, n_jobs=-1),
            True),
    }
    print('  estimating c = P(s=1|y=1) from held-out labelled positives '
          '(folds grouped by sub-basin)...')
    en, helds = {}, {}
    for name, (est, balanced) in estimators.items():
        held = heldout_label_probability(X, y_full, groups, est)
        if balanced:
            held = unbalance(held, n_lab, n_unl)
        helds[name] = held
        c_k = elkan_noto_c(held[y_full == 1])
        pi_k = prior_from_c(n_lab, len(y_full), c_k)
        en[name] = {'c': c_k, 'pi': pi_k}
        print(f'    {name:28s} c = {c_k:.4f}  ->  pi = {pi_k:.5f}')

    c = float(np.nanmedian([v['c'] for v in en.values()]))
    pi_en = prior_from_c(n_lab, len(y_full), c)
    pi_u = pi_unlabelled(pi_en, c)
    print(f'    median c = {c:.4f}   pi = {pi_en:.5f}   pi within U = {pi_u:.5f}')
    print(f'    observed prevalence            = {obs_prev:.5f}')
    if np.isfinite(pi_en) and obs_prev > 0:
        print(f'    ratio pi/observed              = {pi_en/obs_prev:.1f}x')

    km = prior_km(s, y)
    print('  quantile-based upper bounds on pi:')
    for k, v in km.items():
        print(f'    {k}: threshold={v["threshold"]} recall_pos={v["recall_pos"]} '
              f'pi_upper={v["pi_upper"]}')

    auc_pu = auc_rank(y, s)
    auc_corr = corrected_auc(auc_pu, pi_u)
    print(f'  AUC(P vs U) = {auc_pu:.4f}   AUC(P vs N) corrected = {auc_corr:.4f}')
    print(f'    correction = {abs(auc_corr - auc_pu):.5f} (assumes SCAR)')

    boot = bootstrap_c(helds, y_full, s, groups)
    print(f'  sub-basin cluster bootstrap ({boot["n_boot"]} draws, {boot["n_clusters"]} clusters), '
          f'95% CI:')
    for k in ('c', 'pi', 'pi_unlabelled', 'auc_corrected'):
        print(f'    {k:16s} {boot[k]}')

    ops = operating_points(s, y, pi_en)
    print('  operating points (precision under observed vs PU prior):')
    for r in ops:
        print(f'    top {100*r["top_frac"]:5.1f}%  n={r["n"]:6d}  '
              f'caught={r["labelled_caught"]:3d}  '
              f'prec_obs={r["precision_observed"]:.5f}  '
              f'prec_pu={r["precision_pu"]}')

    results = {
        'score': 'oof_score from oof_scores.csv (primary cohort, primary model)',
        'n_lakes': n,
        'n_labelled_positive': n_pos,
        'observed_prevalence': round(obs_prev, 6),
        'elkan_noto': {
            'estimators': {k: {kk: (round(float(vv), 6) if np.isfinite(vv) else None)
                               for kk, vv in v.items()} for k, v in en.items()},
            'c': round(float(c), 5) if np.isfinite(c) else None,
            'pi': round(float(pi_en), 6) if np.isfinite(pi_en) else None,
            'pi_unlabelled': round(float(pi_u), 6) if np.isfinite(pi_u) else None,
            'pi_over_observed': (round(float(pi_en / obs_prev), 2)
                                 if np.isfinite(pi_en) and obs_prev else None),
            'ci95': {k: boot[k] for k in ('c', 'pi', 'pi_unlabelled')},
            'ci_method': f'cluster bootstrap over sub-basins ({boot["n_boot"]} draws, '
                         f'{boot["n_clusters"]} clusters) of the held-out label probabilities; '
                         'estimators are not refitted',
        },
        'quantile_bounds': km,
        'auc': {
            'auc_positive_vs_unlabelled': round(float(auc_pu), 5),
            'auc_corrected': (round(float(auc_corr), 5)
                              if np.isfinite(auc_corr) else None),
            'auc_corrected_ci95': boot['auc_corrected'],
            'note': 'AUC(P vs U) = 0.5*pi_U + (1-pi_U)*AUC(P vs N), with pi_U the '
                    'share of positives within U. Holds only if labelling is '
                    'completely at random (SCAR); a catalogue biased toward '
                    'populated valleys and large events violates it.',
        },
        'operating_points': ops,
    }
    OUT.write_text(json.dumps(results, indent=2))
    print(f'\nsaved: {OUT.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
