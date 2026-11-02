import sys
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

FEATURES = PROJECT_ROOT / 'data' / 'processed' / 'features' / 'lake_features.csv'
PHYS_DIR = PROJECT_ROOT / 'data' / 'processed' / 'physical'
LABEL_DIR = PROJECT_ROOT / 'data' / 'processed' / 'labeled'
LABELS = LABEL_DIR / 'lake_labels.csv'
MATCH_LOG = LABEL_DIR / 'event_match_log.csv'
LAKE_TABLE = LABEL_DIR / 'lake_table.csv'
ALL_EVENTS_TABLE = LABEL_DIR / 'lake_table_all_events.csv'
FORWARD_TABLE = LABEL_DIR / 'lake_table_forward.csv'
FORWARD_TRAIN_TABLE = LABEL_DIR / 'lake_table_forward_train.csv'
MODELS_DIR = PROJECT_ROOT / 'models'
FEATURE_FILE = MODELS_DIR / 'feature_sets.json'
COLLINEARITY_FILE = MODELS_DIR / 'collinearity.csv'

SHAPE_FEATURES = ['area_m2', 'perimeter_m', 'compactness', 'elongation', 'fractal_dim',
                  'convexity', 'shore_dev_index']
DEM_FEATURES = ['lake_elev_m', 'freeboard_m', 'outlet_drop_m', 'ring_slope_mean',
                'ring_slope_p90', 'ring_relief_m', 'ring_aspect_north', 'ring_aspect_east',
                'ring_tri_mean', 'ring_curvature_mean', 'ring_roughness_mean', 'ring_vrm_mean']
GLACIER_FEATURES = ['dist_glacier_m']
BIAS_FEATURES = ['dist_road_m', 'dist_settlement_m']
EXCLUDED_TAGS = ('quarry', 'industrial', 'tailings', 'landfill', 'basin', 'mine')
REGULATED_TAGS = ('dam', 'reservoir')
TEMPORAL_FEATURES = ['area_trend', 'area_cv', 'presence_rate', 'area_last_m2',
                     'years_since_first_seen']
PRIORITY = ['area_m2', 'freeboard_m', 'outlet_drop_m', 'ring_slope_mean', 'dist_glacier_m',
            'area_trend', 'presence_rate', 'lake_elev_m', 'elongation', 'compactness',
            'ring_relief_m', 'ring_aspect_north', 'ring_aspect_east', 'ring_curvature_mean',
            'area_cv', 'years_since_first_seen', 'convexity', 'shore_dev_index',
            'ring_vrm_mean', 'ring_slope_p90', 'ring_tri_mean', 'ring_roughness_mean',
            'fractal_dim', 'perimeter_m', 'area_last_m2']

DEM_YEAR = 2000
FORWARD_YEAR = 2010
END_YEAR = 2025
GROUP_LEVEL = 'basin_l10'
MAX_ABS_RHO = 0.85
MIN_VALID_YEARS = 3
MIN_DETECTIONS = 2
MIN_TREND_YEARS = 3
AMBIGUOUS_DIST_M = 250.0
RANDOM_STATE = 42


def lake_year_table():
    f = pd.read_csv(FEATURES, low_memory=False)
    m = pd.read_csv(PHYS_DIR / 'lake_physical_map.csv', usecols=['lake_id', 'lake_phys', 'acquisition'])
    f = f.drop(columns='acquisition', errors='ignore').merge(m, on='lake_id', how='inner')
    cols = [c for c in SHAPE_FEATURES + DEM_FEATURES + GLACIER_FEATURES if c != 'area_m2']
    agg = {c: 'median' for c in cols}
    agg['area_m2'] = 'sum'
    keys = ['lake_phys', 'year', 'acquisition'] + (['source'] if 'source' in f.columns else [])
    per_acq = f.groupby(keys, as_index=False).agg(agg)
    print(f'  lake-years seen by more than one acquisition or sensor: '
          f'{int(per_acq.duplicated(["lake_phys", "year"]).sum())} (largest area kept)')
    return (per_acq.sort_values('area_m2', ascending=False)
            .drop_duplicates(['lake_phys', 'year'])[['lake_phys', 'year'] + cols + ['area_m2']]
            .sort_values(['lake_phys', 'year']).reset_index(drop=True))


def window_features(ly, val, cutoff):
    cut = cutoff.rename('cutoff')
    ly = ly.join(cut, on='lake_phys', how='inner')
    ly = ly[ly['year'] <= ly['cutoff']]
    val = val.join(cut, on='lake_phys', how='inner')
    val = val[(val['year'] <= val['cutoff']) & (val['valid'] == 1)]
    n_valid = val.groupby('lake_phys')['year'].nunique()

    g = ly.groupby('lake_phys')
    n = g['year'].count()
    cols = [c for c in SHAPE_FEATURES + DEM_FEATURES + GLACIER_FEATURES]
    static = g[cols].median()

    def slope(d):
        if len(d) < MIN_TREND_YEARS:
            return np.nan
        x = d['year'].values.astype(float)
        return float(np.polyfit(x - x.mean(), d['area_m2'].values, 1)[0])

    temporal = pd.DataFrame({
        'area_trend': g.apply(slope, include_groups=False),
        'area_cv': g['area_m2'].std(ddof=0) / g['area_m2'].mean(),
        'presence_rate': n / n_valid.reindex(n.index),
        'area_last_m2': ly.sort_values('year').groupby('lake_phys')['area_m2'].last(),
        'years_since_first_seen': cut.reindex(n.index) - g['year'].min(),
        'n_detections': n,
        'n_valid_years': n_valid.reindex(n.index),
    })
    temporal.loc[n < 2, 'area_cv'] = np.nan
    return static.join(temporal)


def eligible(df):
    return (df['n_valid_years'] >= MIN_VALID_YEARS) & (df['n_detections'] >= MIN_DETECTIONS)


def lake_info():
    m = pd.read_csv(PHYS_DIR / 'lake_physical_map.csv')
    extra = [c for c in ['anthropogenic', 'anthropogenic_tag'] + BIAS_FEATURES if c in m.columns]
    keep = ['lake_phys', 'analysis_region', 'country', 'lon', 'lat'] + \
        [c for c in m.columns if c.startswith('basin_l')] + extra
    info = m[keep].drop_duplicates('lake_phys').set_index('lake_phys')
    info['group'] = info[GROUP_LEVEL]
    if 'anthropogenic' not in info.columns:
        print('  [WARN] no anthropogenic flag: run 00_osm_context.py before 06_physical_lakes.py')
        info['anthropogenic'] = False
    for c in BIAS_FEATURES:
        if c not in info.columns:
            info[c] = np.nan
    info['anthropogenic'] = info['anthropogenic'].fillna(False).astype(bool)
    tag = info.get('anthropogenic_tag', pd.Series('', index=info.index)).fillna('').str.lower()
    info['industrial'] = info['anthropogenic'] & tag.apply(lambda t: any(k in t for k in EXCLUDED_TAGS))
    info['regulated'] = info['anthropogenic'] & tag.apply(lambda t: any(k in t for k in REGULATED_TAGS))
    return info


def match_quality():
    log = pd.read_csv(MATCH_LOG)
    log = log[log['status'] == 'matched']
    return log.groupby('lake_phys').agg(match_distance_m=('distance_m', 'min'),
                                        match_candidates=('n_within', 'max'))


def collinearity(lakes, features):
    x = lakes[features].rank()
    rho = x.corr().abs()
    order = [f for f in PRIORITY if f in features] + [f for f in features if f not in PRIORITY]
    kept, rows = [], []
    for f in order:
        clash = [k for k in kept if rho.at[f, k] > MAX_ABS_RHO]
        if clash:
            rows.append({'feature': f, 'dropped': True, 'because_of': clash[0],
                         'abs_spearman': round(float(rho.at[f, clash[0]]), 3)})
        else:
            kept.append(f)
            rows.append({'feature': f, 'dropped': False, 'because_of': '', 'abs_spearman': np.nan})
    return kept, pd.DataFrame(rows)


def vif(df):
    from numpy.linalg import lstsq
    z = df.fillna(df.median())
    z = (z - z.mean()) / z.std(ddof=0).replace(0, 1)
    out = {}
    for c in z.columns:
        others = z.drop(columns=c).values
        a = np.c_[np.ones(len(z)), others]
        coef = lstsq(a, z[c].values, rcond=None)[0]
        r2 = 1 - np.var(z[c].values - a @ coef) / max(np.var(z[c].values), 1e-12)
        out[c] = 1 / max(1 - r2, 1e-6)
    return pd.Series(out)


def build_window(ly, val, labels, info, first_year=None, last_year=None, verbose=False):
    rng = np.random.default_rng(RANDOM_STATE)
    lakes = labels.join(info, how='inner')
    drop = lakes['industrial'] & (lakes['label'] == 0)
    if verbose:
        print(f'  mining or industrial water bodies removed: {int(drop.sum())}; '
              f'positives on such features kept for review: '
              f'{int((lakes["industrial"] & (lakes["label"] == 1)).sum())}; '
              f'regulated lakes (dam or reservoir) kept: {int(lakes["regulated"].sum())}')
    lakes = lakes[~drop].copy()
    if last_year is not None:
        unknown = (lakes['label'] == 1) & (lakes['event_year'] > last_year)
        lakes.loc[unknown, 'label'] = 0
        lakes.loc[unknown, 'event_year'] = np.nan
        lakes.loc[unknown, 'label_status'] = 'event_after_window'
    if first_year is not None:
        lakes = lakes[~((lakes['label'] == 1) & (lakes['event_year'] <= first_year))].copy()
    pos_years = lakes.loc[lakes['label'] == 1, 'event_year'].astype(int).values
    if len(pos_years) == 0:
        return None
    cutoff = lakes['event_year'] - 1
    neg = lakes['label'] == 0
    cutoff[neg] = rng.choice(pos_years, size=int(neg.sum())) - 1
    lakes['cutoff'] = cutoff.astype(int)
    lakes = lakes.join(window_features(ly, val, lakes['cutoff']), how='inner')
    before = lakes['label'].value_counts().to_dict()
    lakes = lakes[eligible(lakes)].copy()
    after = lakes['label'].value_counts().to_dict()
    if verbose:
        print(f'  eligibility (>= {MIN_VALID_YEARS} valid years and >= {MIN_DETECTIONS} detections '
              f'before cutoff, same rule for both classes): positives {before.get(1, 0)} -> '
              f'{after.get(1, 0)}, negatives {before.get(0, 0)} -> {after.get(0, 0)}')
    lakes = lakes.join(match_quality(), how='left')
    lakes['dem_before_event'] = (lakes['label'] == 0) | (lakes['event_year'] > DEM_YEAR)
    lakes['ambiguous_match'] = (lakes['label'] == 1) & (
        (lakes['match_candidates'] > 1) | (lakes['match_distance_m'] > AMBIGUOUS_DIST_M))
    return lakes


def build_forward(ly, val, labels, info):
    lakes = labels.join(info, how='inner')
    lakes = lakes[~(lakes['industrial'] & (lakes['label'] == 0))]
    lakes = lakes[~((lakes['label'] == 1) & (lakes['event_year'] <= FORWARD_YEAR))].copy()
    lakes['cutoff'] = FORWARD_YEAR
    lakes = lakes.join(window_features(ly, val, lakes['cutoff']), how='inner')
    lakes = lakes[eligible(lakes)].copy()
    lakes['label'] = ((lakes['label'] == 1) & (lakes['event_year'] > FORWARD_YEAR)
                      & (lakes['event_year'] <= END_YEAR)).astype(int)
    return lakes


def cutoff_summary(lakes):
    c = lakes.groupby('label')['cutoff'].describe()[['min', '50%', 'max']].astype(int)
    return ', '.join(f'{"pos" if k == 1 else "neg"} {r["min"]}/{r["50%"]}/{r["max"]}'
                     for k, r in c.iterrows())


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS).set_index('lake_phys')
    labels = labels[labels['label'] >= 0]
    info = lake_info()
    ly = lake_year_table()
    val = pd.read_csv(PHYS_DIR / 'lake_year_validity.csv')

    lakes = build_window(ly, val, labels, info, first_year=DEM_YEAR, verbose=True)
    if lakes is None:
        sys.exit(f'[ERROR] no positive with an event after {DEM_YEAR}')
    lakes_all = build_window(ly, val, labels, info)
    for name, t in (('DEM-era', lakes), ('all-events', lakes_all)):
        print(f'  {name} table: lakes {len(t)} positives {int(t["label"].sum())}  '
              f'cutoff min/median/max {cutoff_summary(t)}')

    primary = lakes[lakes['dem_before_event']]
    candidates = SHAPE_FEATURES + DEM_FEATURES + GLACIER_FEATURES + TEMPORAL_FEATURES
    kept, table = collinearity(primary, candidates)
    v = vif(primary[kept])
    table['vif'] = table['feature'].map(v).round(2)
    table.to_csv(COLLINEARITY_FILE, index=False)
    print(f'  collinearity on the primary cohort ({len(primary)} lakes) |rho| > {MAX_ABS_RHO}: '
          f'dropped {table.loc[table["dropped"], "feature"].tolist()}')
    print(f'  max VIF of kept features: {v.max():.1f} ({v.idxmax()})')

    sets = {
        'full': kept,
        'no_dem': [f for f in kept if f not in DEM_FEATURES],
        'static': [f for f in kept if f not in TEMPORAL_FEATURES],
    }
    bias = [c for c in BIAS_FEATURES if lakes[c].notna().any()]
    FEATURE_FILE.write_text(json.dumps({'feature_sets': sets, 'dem_features': DEM_FEATURES,
                                        'bias_features': bias,
                                        'group_level': GROUP_LEVEL, 'dem_year': DEM_YEAR,
                                        'forward_year': FORWARD_YEAR}, indent=2))
    lakes.reset_index().to_csv(LAKE_TABLE, index=False)
    lakes_all.reset_index().to_csv(ALL_EVENTS_TABLE, index=False)

    pos = lakes_all[lakes_all['label'] == 1]
    print(f'  lakes {len(lakes_all)}  positives {len(pos)}  groups {lakes_all["group"].nunique()}')
    print(f'  positives with DEM before event (> {DEM_YEAR}): {int(pos["dem_before_event"].sum())}'
          f'  ambiguous matches: {int(pos["ambiguous_match"].sum())}')
    print(pos.groupby(['country', 'analysis_region']).size().to_string())

    train = []
    for window, first in (('dem', DEM_YEAR), ('all', None)):
        t = build_window(ly, val, labels, info, first_year=first, last_year=FORWARD_YEAR)
        if t is None:
            print(f'  [note] forward training window {window}: no positive up to {FORWARD_YEAR}')
            continue
        train.append(t.assign(window=window))
        print(f'  forward training window {window} (events <= {FORWARD_YEAR}): lakes {len(t)} '
              f'positives {int(t["label"].sum())}')
    if train:
        pd.concat(train).reset_index().to_csv(FORWARD_TRAIN_TABLE, index=False)
    fwd = build_forward(ly, val, labels, info)
    fwd.reset_index().to_csv(FORWARD_TABLE, index=False)
    print(f'  forward table (features at {FORWARD_YEAR}, events {FORWARD_YEAR + 1}-{END_YEAR}): '
          f'lakes {len(fwd)}  positives {int(fwd["label"].sum())}')
    print(f'  saved {LAKE_TABLE.name}, {ALL_EVENTS_TABLE.name}, {FORWARD_TRAIN_TABLE.name}, '
          f'{FORWARD_TABLE.name}, {FEATURE_FILE.name}, {COLLINEARITY_FILE.name}')


if __name__ == '__main__':
    main()
