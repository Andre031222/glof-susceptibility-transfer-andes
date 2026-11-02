import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import EXPANDED_STUDY_AREAS

EVENTS = PROJECT_ROOT / 'data' / 'source' / 'historical_glofs.gpkg'
PHYS_DIR = PROJECT_ROOT / 'data' / 'processed' / 'physical'
FOOTPRINTS = PHYS_DIR / 'lake_footprints.gpkg'
OUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'labeled'
LABELS = OUT_DIR / 'lake_labels.csv'
LOG = OUT_DIR / 'event_match_log.csv'

MODELLED_DAMS = {'moraine', 'bedrock'}
FIRST_IMAGE_YEAR = 1985


def utm_epsg(lon, lat):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def match_events(ev, fp, max_dist):
    rows = []
    for _, e in ev.iterrows():
        epsg = utm_epsg(e.geometry.x, e.geometry.y)
        pt = gpd.GeoSeries([e.geometry], crs=4326).to_crs(epsg).iloc[0]
        cand = fp[(fp['lon'] - e.geometry.x).abs() < 0.1]
        cand = cand[(cand['lat'] - e.geometry.y).abs() < 0.1]
        rec = {'event_id': e['event_id'], 'lake': e['lake'], 'year': int(e['year']),
               'lake_type': e['lake_type'], 'lon': e.geometry.x, 'lat': e.geometry.y,
               'lake_phys': None, 'distance_m': np.nan, 'n_within': 0}
        inside = any(b[0] <= e.geometry.x <= b[2] and b[1] <= e.geometry.y <= b[3]
                     for b in (c['bbox'] for c in EXPANDED_STUDY_AREAS.values()))
        if cand.empty:
            rec['status'] = 'no_lake_nearby' if inside else 'outside_study_areas'
            rows.append(rec)
            continue
        d = cand.to_crs(epsg).geometry.distance(pt)
        best = d.groupby(cand['lake_phys']).min().sort_values()
        rec['distance_m'] = float(best.iloc[0])
        rec['n_within'] = int((best <= max_dist).sum())
        if best.iloc[0] <= max_dist:
            rec['lake_phys'] = best.index[0]
            rec['status'] = 'matched'
        else:
            rec['status'] = 'too_far'
        rows.append(rec)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-dist', type=float, default=500.0)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev = gpd.read_file(EVENTS).to_crs(4326)
    fp = gpd.read_file(FOOTPRINTS).to_crs(4326)
    print(f'  events: {len(ev)}  physical lakes: {fp["lake_phys"].nunique()}')

    log = match_events(ev, fp, args.max_dist)
    log.to_csv(LOG, index=False)
    print(log['status'].value_counts().to_string())

    m = log[log['status'] == 'matched']
    lakes = pd.DataFrame({'lake_phys': sorted(fp['lake_phys'].unique())})
    lakes['label'] = 0
    lakes['label_status'] = 'unlabelled'
    lakes['event_year'] = np.nan
    lakes['dam_type'] = ''
    lakes['n_events'] = 0
    lakes = lakes.set_index('lake_phys')

    for lp, d in m.groupby('lake_phys'):
        types = set(d['lake_type'])
        later = d[(d['year'] > FIRST_IMAGE_YEAR + 1) & d['lake_type'].isin(MODELLED_DAMS)]
        lakes.at[lp, 'n_events'] = len(d)
        lakes.at[lp, 'dam_type'] = '/'.join(sorted(types))
        if not types & MODELLED_DAMS:
            lakes.at[lp, 'label'] = -1
            lakes.at[lp, 'label_status'] = 'excluded_dam_type'
        elif later.empty:
            lakes.at[lp, 'label'] = -1
            lakes.at[lp, 'label_status'] = 'excluded_pre_record'
        else:
            lakes.at[lp, 'label'] = 1
            lakes.at[lp, 'label_status'] = 'positive'
            lakes.at[lp, 'event_year'] = int(later['year'].min())

    lakes = lakes.reset_index()
    lakes.to_csv(LABELS, index=False)
    print(lakes['label_status'].value_counts().to_string())
    amb = m[m['n_within'] > 1]
    if len(amb):
        print(f'  [note] {len(amb)} events had more than one lake within {args.max_dist:.0f} m; '
              f'nearest taken')
    print(f'  saved {LABELS.name}, {LOG.name}')


if __name__ == '__main__':
    main()
