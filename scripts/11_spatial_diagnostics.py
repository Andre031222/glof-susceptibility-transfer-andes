import os
import sys
import json
import argparse
import importlib.util
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.neighbors import BallTree

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = Path(os.environ.get('GLOF_ROOT') or SCRIPTS_DIR.parent)
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

MODELS_DIR = PROJECT_ROOT / 'models'
LAKE_TABLE = PROJECT_ROOT / 'data' / 'processed' / 'labeled' / 'lake_table.csv'
INPUTS = {
    'oof': MODELS_DIR / 'oof_scores.csv',
    'features': MODELS_DIR / 'feature_sets.json',
    'run': MODELS_DIR / 'run_summary.json',
    'calibrated': MODELS_DIR / 'oof_calibrated.csv',
}
OUTPUTS = {
    'decay_blocks': MODELS_DIR / 'distance_decay_blocks.csv',
    'decay_bins': MODELS_DIR / 'distance_decay_bins.csv',
    'correlogram': MODELS_DIR / 'spatial_autocorrelation.csv',
    'aoa_region': MODELS_DIR / 'aoa_by_region.csv',
    'di_lakes': MODELS_DIR / 'di_lakes.csv',
    'summary': MODELS_DIR / 'spatial_summary.json',
}

EARTH_RADIUS_KM = 6371.0088
EQUAL_AREA_CRS = 'ESRI:102033'
RANDOM_STATE = 42
BLOCKS_KM = [10, 25, 50, 100, 200]
N_OFFSETS = 5
DIST_BINS_KM = [0, 1, 2, 5, 10, 25, 50, 100, 250, np.inf]
BANDS_KM = [(0, 2), (2, 5), (5, 10), (10, 25), (25, 50), (50, 100), (100, 200)]
CHUNK = 256
P_EPS = 1e-3
N_CAL_SPLITS = 5
DI_SAMPLE = 3000
MIN_BIN_POS = 3


def load_train_module():
    spec = importlib.util.spec_from_file_location('train10',
                                                  str(SCRIPTS_DIR / '10_train_validate.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def require(paths):
    for f in paths:
        if not f.exists():
            sys.exit(f'[ERROR] {f} not found; run 08_lake_table.py and 10_train_validate.py first')


def calibrate_oof(y, s):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.isotonic import IsotonicRegression
    cal = np.full(len(y), np.nan)
    idx = np.flatnonzero(np.isfinite(s))
    skf = StratifiedKFold(N_CAL_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    for tr, te in skf.split(idx.reshape(-1, 1), y[idx]):
        tr, te = idx[tr], idx[te]
        ir = IsotonicRegression(out_of_bounds='clip', y_min=0.0, y_max=1.0).fit(s[tr], y[tr])
        cal[te] = ir.predict(s[te])
    return np.nan_to_num(cal, nan=float(y[idx].mean()))


def load_inputs(m):
    require([LAKE_TABLE, INPUTS['oof'], INPUTS['features'], INPUTS['run']])
    cfg = json.loads(INPUTS['features'].read_text())
    run = json.loads(INPUTS['run'].read_text())
    features = cfg['feature_sets'][run.get('primary_feature_set', m.PRIMARY_SET)]
    lakes_all = pd.read_csv(LAKE_TABLE, low_memory=False)
    lakes = m.cohort_table(lakes_all, m.COHORTS['primary'])
    oof = pd.read_csv(INPUTS['oof']).set_index('lake_key')
    if set(oof.index) != set(lakes['lake_phys']):
        raise ValueError('oof_scores.csv and the primary cohort of lake_table.csv differ; re-run 10')
    oof = oof.reindex(lakes['lake_phys'])
    if not np.array_equal(oof['glof'].values.astype(int), lakes['label'].values.astype(int)):
        raise ValueError('labels differ between oof_scores.csv and lake_table.csv; re-run 10')
    lakes['oof_score'] = oof['oof_score'].values
    lakes['oof_region'] = oof['oof_region'].values if 'oof_region' in oof else np.nan
    y = lakes['label'].values.astype(int)
    s = lakes['oof_score'].values.astype(float)
    source = 'cross-fitted isotonic inside 11 (oof_calibrated.csv missing or stale)'
    cal = None
    if INPUTS['calibrated'].exists():
        c = pd.read_csv(INPUTS['calibrated']).set_index('lake_key').reindex(lakes['lake_phys'])
        if (c['calibrated'].notna().all()
                and np.allclose(c['oof_score'].values, s, atol=1e-6, equal_nan=True)):
            cal = c['calibrated'].values.astype(float)
            source = 'oof_calibrated.csv (12_paper_stats.py)'
    if cal is None:
        cal = calibrate_oof(y, s)
    lakes['calibrated'] = cal
    model_name = run.get('primary_model', m.PRIMARY_MODEL)
    return lakes, features, model_name, source


def haversine_tree(lon, lat):
    return BallTree(np.radians(np.c_[lat, lon]), metric='haversine')


def nearest_km(lon, lat, query, data):
    tree = haversine_tree(lon[data], lat[data])
    d, _ = tree.query(np.radians(np.c_[lat[query], lon[query]]), k=1)
    return d[:, 0] * EARTH_RADIUS_KM


def projected(lon, lat):
    from pyproj import Transformer
    t = Transformer.from_crs(4326, EQUAL_AREA_CRS, always_xy=True)
    x, y = t.transform(lon, lat)
    return np.asarray(x), np.asarray(y)


def block_ids(x, y, km, offset):
    size = km * 1000.0
    bx = np.floor((x + offset[0] * size) / size).astype(np.int64)
    by = np.floor((y + offset[1] * size) / size).astype(np.int64)
    return bx * 1_000_003 + by


def decay_designs(lakes, y, x_m, y_m, n_offsets):
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
    rng = np.random.default_rng(RANDOM_STATE)
    idx = np.arange(len(y))
    designs = []
    for r in range(n_offsets):
        seed = RANDOM_STATE + r
        designs.append(('random', 0.0, r, list(
            StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, y))))
        designs.append(('group', 0.0, r, list(
            StratifiedGroupKFold(5, shuffle=True, random_state=seed)
            .split(idx, y, lakes['group'].values))))
    for km in BLOCKS_KM:
        for r in range(n_offsets):
            off = rng.uniform(0, 1, 2) if r else np.zeros(2)
            blocks = block_ids(x_m, y_m, km, off)
            k = min(5, len(np.unique(blocks[y == 1])))
            if k < 2:
                print(f'  [skip] {km} km offset {r}: {k} blocks hold positives')
                continue
            designs.append(('block', float(km), r, list(
                StratifiedGroupKFold(k, shuffle=True, random_state=RANDOM_STATE + r)
                .split(idx, y, blocks))))
    return designs


def distance_decay(m, lakes, features, model_name, n_offsets, n_jobs):
    model = m.build_models(n_jobs=1)[model_name]
    X = lakes[features].values.astype(float)
    y = lakes['label'].values.astype(int)
    lon, lat = lakes['lon'].values.astype(float), lakes['lat'].values.astype(float)
    x_m, y_m = projected(lon, lat)
    designs = decay_designs(lakes, y, x_m, y_m, n_offsets)
    tasks = [(d, tr, te) for d, (_, _, _, sp) in enumerate(designs) for tr, te in sp]
    print(f'  {len(designs)} partitions, {len(tasks)} model fits ({model_name}, '
          f'{len(features)} features)')
    fits = Parallel(n_jobs=n_jobs)(delayed(m.run_fold)(model, X, y, tr, te)
                                   for _, tr, te in tasks)
    oof = np.full((len(designs), len(y)), np.nan)
    sep = np.full((len(designs), len(y)), np.nan)
    for (d, tr, te), (_, s) in zip(tasks, fits):
        oof[d, te] = s
        sep[d, te] = nearest_km(lon, lat, te, tr)
    rows, pooled = [], []
    for d, (scheme, km, r, sp) in enumerate(designs):
        ok = np.isfinite(oof[d])
        met = m.metrics(y[ok], oof[d, ok])
        rows.append({'scheme': scheme, 'block_km': km, 'offset': r, 'n_splits': len(sp),
                     'n_scored': int(ok.sum()), 'n_pos': int(y[ok].sum()),
                     'median_separation_km': float(np.nanmedian(sep[d])),
                     'median_separation_pos_km': float(np.nanmedian(sep[d][y == 1])), **met})
        pooled.append(pd.DataFrame({'scheme': scheme, 'y': y[ok], 's': oof[d, ok],
                                    'sep': sep[d, ok]}))
    blocks = pd.DataFrame(rows)
    for (scheme, km), g in blocks.groupby(['scheme', 'block_km']):
        tag = f'{km:.0f} km' if scheme == 'block' else ''
        print(f'    {scheme:6s} {tag:7s} AUC={g["auc"].mean():.3f} '
              f'(sd {g["auc"].std(ddof=0):.3f}, {len(g)} offsets)  '
              f'separation median {g["median_separation_km"].median():.1f} km')
    return blocks, by_distance(pd.concat(pooled, ignore_index=True))


def by_distance(pool):
    from sklearn.metrics import roc_auc_score
    rows = []
    for lo, hi in zip(DIST_BINS_KM[:-1], DIST_BINS_KM[1:]):
        b = pool[(pool['sep'] >= lo) & (pool['sep'] < hi)]
        npos = int(b['y'].sum())
        auc = (float(roc_auc_score(b['y'], b['s']))
               if MIN_BIN_POS <= npos < len(b) else np.nan)
        rows.append({'dist_lo_km': lo, 'dist_hi_km': hi, 'n_predictions': int(len(b)),
                     'n_pos': npos, 'auc': auc})
        if len(b):
            print(f'    {lo:>5}-{hi:<5} km  n={len(b):7d}  pos={npos:4d}  AUC={auc:.3f}')
    return pd.DataFrame(rows)


def band_statistics(lon, lat, Z):
    n, k = Z.shape
    edges = np.array([BANDS_KM[0][0]] + [hi for _, hi in BANDS_KM], dtype=float)
    nb = len(BANDS_KM)
    tree = haversine_tree(lon, lat)
    pts = np.radians(np.c_[lat, lon])
    r = edges[-1] / EARTH_RADIUS_KM
    deg = np.zeros((nb, n))
    cross = np.zeros((nb, k))
    for start in range(0, n, CHUNK):
        idx = np.arange(start, min(start + CHUNK, n))
        nbrs, dist = tree.query_radius(pts[idx], r=r, return_distance=True)
        sizes = np.array([len(a) for a in nbrs])
        if not sizes.sum():
            continue
        rows = np.repeat(idx, sizes)
        cols = np.concatenate(nbrs)
        d_km = np.concatenate(dist) * EARTH_RADIUS_KM
        keep = rows != cols
        rows, cols, d_km = rows[keep], cols[keep], d_km[keep]
        band = np.searchsorted(edges, d_km, side='right') - 1
        valid = (band >= 0) & (band < nb)
        for b in range(nb):
            sel = valid & (band == b)
            if not sel.any():
                continue
            rb, cb = rows[sel], cols[sel]
            deg[b] += np.bincount(rb, minlength=n)
            cross[b] += np.einsum('ij,ij->j', Z[rb], Z[cb])
    return deg, cross


def moran_randomization(z, cross_sum, deg):
    n = len(z)
    s0 = deg.sum()
    m2 = (z ** 2).sum()
    if s0 == 0 or n < 4 or m2 == 0:
        return np.nan, np.nan, np.nan, np.nan
    obs = n / s0 * cross_sum / m2
    s1 = 2.0 * s0
    s2 = float((4.0 * deg ** 2).sum())
    e = -1.0 / (n - 1)
    b2 = n * (z ** 4).sum() / m2 ** 2
    num = (n * ((n * n - 3 * n + 3) * s1 - n * s2 + 3 * s0 ** 2)
           - b2 * ((n * n - n) * s1 - 2 * n * s2 + 6 * s0 ** 2))
    var = num / ((n - 1) * (n - 2) * (n - 3) * s0 ** 2) - e ** 2
    if var <= 0:
        return float(obs), e, np.nan, np.nan
    zs = (obs - e) / np.sqrt(var)
    return float(obs), e, float(zs), float(2 * norm.sf(abs(zs)))


def correlogram(lon, lat, variables, kinds):
    names = list(variables)
    cols = []
    for v in names:
        a = np.asarray(variables[v], dtype=float)
        a = np.where(np.isfinite(a), a, np.nanmedian(a))
        cols.append(a - a.mean())
    Z = np.column_stack(cols)
    deg, cross = band_statistics(lon, lat, Z)
    rows = []
    for b, (lo, hi) in enumerate(BANDS_KM):
        pairs = int(deg[b].sum() // 2)
        for j, name in enumerate(names):
            obs, e, zs, p = moran_randomization(Z[:, j], cross[b, j], deg[b])
            rows.append({'variable': name, 'kind': kinds[name], 'band_lo_km': lo,
                         'band_hi_km': hi, 'n_pairs': pairs,
                         'mean_neighbours': float(deg[b].mean()), 'morans_I': obs,
                         'expected_I': e, 'z_score': zs, 'p_value': p})
    return pd.DataFrame(rows)


def autocorrelation_range(corr, variable):
    res = corr[corr['variable'] == variable].sort_values('band_hi_km')
    for _, r in res.iterrows():
        if not np.isfinite(r['p_value']) or r['p_value'] >= 0.05 or r['morans_I'] <= 0:
            return float(r['band_lo_km'])
    return float(res['band_hi_km'].max())


def min_dist(A, B):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=1).fit(B)
    return nn.kneighbors(A, return_distance=True)[0][:, 0]


def upper_whisker(v):
    v = v[np.isfinite(v)]
    q1, q3 = np.percentile(v, [25, 75])
    lim = q3 + 1.5 * (q3 - q1)
    return float(v[v <= lim].max())


def dissimilarity(m, model, X_tr, y_tr, g_tr, r_tr, X_te, rng):
    from sklearn.model_selection import StratifiedGroupKFold
    pipe = m.make_pipeline(model, y_tr).fit(X_tr, y_tr)
    imp = pipe.named_steps['imputer']
    weights = np.asarray(pipe.named_steps['model'].feature_importances_, dtype=float)
    A = imp.transform(X_tr)
    B = imp.transform(X_te)
    mu, sd = A.mean(0), A.std(0)
    sd[sd == 0] = 1.0
    A = (A - mu) / sd * weights
    B = (B - mu) / sd * weights
    sub = A[rng.choice(len(A), min(DI_SAMPLE, len(A)), replace=False)]
    sq = (sub ** 2).sum(1)
    d = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2 * sub @ sub.T, 0))
    mean_d = float(d[np.triu_indices(len(sub), 1)].mean())
    train_di = np.full(len(A), np.nan)
    regions = np.unique(r_tr)
    if len(regions) >= 2:
        folds = [(np.flatnonzero(r_tr != r), np.flatnonzero(r_tr == r)) for r in regions]
        cv = 'leave-one-region-out'
    else:
        k = min(5, len(np.unique(g_tr[y_tr == 1])))
        folds = StratifiedGroupKFold(max(k, 2), shuffle=True, random_state=RANDOM_STATE) \
            .split(A, y_tr, g_tr)
        cv = 'sub-basin grouped'
    for tr, te in folds:
        train_di[te] = min_dist(A[te], A[tr]) / mean_d
    threshold = upper_whisker(train_di)
    return min_dist(B, A) / mean_d, threshold, weights, cv


def aoa_by_region(m, lakes, features, model_name):
    from sklearn.metrics import roc_auc_score
    model = m.build_models(n_jobs=-1)[model_name]
    X = lakes[features].values.astype(float)
    y = lakes['label'].values.astype(int)
    g = lakes['group'].values
    reg = lakes['analysis_region'].values
    rng = np.random.default_rng(RANDOM_STATE)
    rows, lake_rows = [], []
    for r in sorted(np.unique(reg)):
        te = reg == r
        tr = ~te
        if y[tr].sum() < 2:
            print(f'  [skip] {r}: training set has fewer than 2 positives')
            continue
        di, thr, w, cv = dissimilarity(m, model, X[tr], y[tr], g[tr], reg[tr], X[te], rng)
        inside = di <= thr
        s = lakes.loc[te, 'oof_region'].values.astype(float)
        yt = y[te]

        def auc(mask):
            ok = mask & np.isfinite(s)
            return (float(roc_auc_score(yt[ok], s[ok]))
                    if 0 < yt[ok].sum() < ok.sum() else np.nan)

        rows.append({'held_out': r, 'n_test': int(te.sum()), 'n_pos': int(yt.sum()),
                     'di_threshold': thr, 'threshold_cv': cv,
                     'di_median': float(np.median(di)),
                     'frac_inside_aoa': float(inside.mean()),
                     'n_pos_inside_aoa': int(yt[inside].sum()),
                     'auc_inside_aoa': auc(inside), 'auc_outside_aoa': auc(~inside),
                     'top_weight_feature': features[int(np.argmax(w))]})
        lake_rows.append(pd.DataFrame({
            'lake_phys': lakes.loc[te, 'lake_phys'].values, 'analysis_region': r,
            'country': lakes.loc[te, 'country'].values, 'lon': lakes.loc[te, 'lon'].values,
            'lat': lakes.loc[te, 'lat'].values, 'glof': yt, 'di': np.round(di, 5),
            'di_threshold': round(thr, 5), 'inside_aoa': inside, 'oof_region': s}))
        print(f'    {r:26s} n={te.sum():6d} pos={yt.sum():3d}  DI median {np.median(di):.3f} '
              f'threshold {thr:.3f}  inside AOA {100 * inside.mean():5.1f}%')
    lake_df = (pd.concat(lake_rows, ignore_index=True) if lake_rows
               else pd.DataFrame(columns=['lake_phys', 'analysis_region', 'country', 'lon', 'lat',
                                          'glof', 'di', 'di_threshold', 'inside_aoa',
                                          'oof_region']))
    return pd.DataFrame(rows), lake_df


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
    parser = argparse.ArgumentParser()
    parser.add_argument('--offsets', type=int, default=N_OFFSETS)
    parser.add_argument('--jobs', type=int, default=-1)
    args = parser.parse_args()

    print('=== Spatial diagnostics ===')
    m = load_train_module()
    lakes, features, model_name, cal_source = load_inputs(m)
    y = lakes['label'].values.astype(float)
    lon, lat = lakes['lon'].values.astype(float), lakes['lat'].values.astype(float)
    print(f'  primary cohort: lakes={len(y)}  positives={int(y.sum())}  model={model_name}  '
          f'features={len(features)}')

    print('\n1. Distance decay: AUC against block size and against distance to training')
    blocks, bins = distance_decay(m, lakes, features, model_name, args.offsets, args.jobs)
    blocks.to_csv(OUTPUTS['decay_blocks'], index=False)
    bins.to_csv(OUTPUTS['decay_bins'], index=False)

    print(f'\n2. Moran correlogram (haversine bands, randomisation moments); calibration: '
          f'{cal_source}')
    p = lakes['calibrated'].values.astype(float)
    pc = np.clip(p, P_EPS, 1 - P_EPS)
    variables = {'glof': y, 'oof_score': lakes['oof_score'].values, 'calibrated': p,
                 'residual': y - p, 'pearson_residual': (y - pc) / np.sqrt(pc * (1 - pc))}
    kinds = {'glof': 'response', 'oof_score': 'score', 'calibrated': 'score',
             'residual': 'residual', 'pearson_residual': 'residual'}
    for f in features:
        variables[f] = lakes[f].values
        kinds[f] = 'predictor'
    corr = correlogram(lon, lat, variables, kinds)
    corr.to_csv(OUTPUTS['correlogram'], index=False)
    ranges = {v: autocorrelation_range(corr, v) for v in variables}
    for v in variables:
        print(f'    {v:24s} {kinds[v]:9s} positive autocorrelation up to ~{ranges[v]:.0f} km')

    print('\n3. Dissimilarity index and area of applicability, leave-one-region-out')
    aoa, di = aoa_by_region(m, lakes, features, model_name)
    aoa.to_csv(OUTPUTS['aoa_region'], index=False)
    di.to_csv(OUTPUTS['di_lakes'], index=False)

    decay = blocks.groupby(['scheme', 'block_km']).agg(
        auc_mean=('auc', 'mean'), auc_sd=('auc', 'std'), n_offsets=('auc', 'size'),
        median_separation_km=('median_separation_km', 'median')).reset_index()
    summary = {
        'primary_model': model_name, 'n_lakes': int(len(y)), 'n_pos': int(y.sum()),
        'features': features, 'distance': 'great-circle (haversine, BallTree)',
        'blocks_km': BLOCKS_KM, 'n_offsets': args.offsets,
        'block_grid': f'{EQUAL_AREA_CRS} square grid with random offsets',
        'distance_decay': decay.to_dict(orient='records'),
        'moran_inference': 'analytical moments under randomisation (Cliff and Ord)',
        'calibration_source': cal_source, 'bands_km': BANDS_KM,
        'autocorrelation_range_km': ranges,
        'residual_range_km': ranges['residual'],
        'predictor_range_km_median': float(np.median([ranges[f] for f in features])),
        'aoa_method': 'Meyer and Pebesma (2021): importance-weighted standardised predictors, '
                      'DI = nearest training distance / mean training distance, threshold = '
                      'upper whisker of training DI under the folds of the assessed CV '
                      '(leave-one-region-out among the training regions; sub-basin grouped '
                      'folds when only one region trains); weights = impurity importance of '
                      'the leave-one-region-out model',
        'aoa_pooled_fraction_inside': float(di['inside_aoa'].mean()) if len(di) else None,
        'outputs': {k: v.name for k, v in OUTPUTS.items()},
    }
    OUTPUTS['summary'].write_text(json.dumps(clean_json(summary), indent=2, allow_nan=False))
    print('\n  saved ' + ', '.join(v.name for v in OUTPUTS.values()))


if __name__ == '__main__':
    main()
