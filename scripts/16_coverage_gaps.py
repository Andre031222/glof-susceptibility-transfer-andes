import os
import sys
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(os.environ.get('GLOF_ROOT') or Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_expanded_study_areas import EXPANDED_STUDY_AREAS, analysis_region

RAW_DIR = PROJECT_ROOT / 'data' / 'raw'
GLOFS = PROJECT_ROOT / 'data' / 'source' / 'historical_glofs.csv'
OUT = PROJECT_ROOT / 'models' / 'download_plan.json'

SENSORS = {
    'landsat': {'dir': 'landsat', 'years': range(1985, 2017), 'event_window': 2},
    'sentinel2': {'dir': 'sentinel2', 'years': range(2017, 2026), 'event_window': 1},
}
REQUIRED = ('mndwi.tif', 'ndwi.tif', 'nir.tif')
N_YEARS = sum(len(s['years']) for s in SENSORS.values())


def year_status(sensor, area, year):
    d = RAW_DIR / SENSORS[sensor]['dir'] / area / str(year)
    present = [f for f in REQUIRED if (d / f).exists()]
    if len(present) == len(REQUIRED):
        return 'complete'
    return 'partial' if present else 'missing'


def main():
    glofs = pd.read_csv(GLOFS)
    first = min(SENSORS['landsat']['years'])

    plan = []
    print(f'=== coverage gaps per acquisition area (required: {", ".join(REQUIRED)}) ===\n')
    for area, cfg in EXPANDED_STUDY_AREAS.items():
        status = {s: {y: year_status(s, area, y) for y in SENSORS[s]['years']} for s in SENSORS}
        missing = {s: [y for y, v in st.items() if v != 'complete'] for s, st in status.items()}
        partial = {s: [y for y, v in st.items() if v == 'partial'] for s, st in status.items()}

        bb = cfg['bbox']
        inside = glofs[glofs['lon'].between(bb[0], bb[2]) & glofs['lat'].between(bb[1], bb[3])]
        years = pd.to_numeric(inside['year'], errors='coerce').dropna().astype(int)
        ev_years = sorted(int(y) for y in years if y >= first)
        n_pre = int((years < first).sum())
        blocking = []
        for y in ev_years:
            gaps = [m for s, spec in SENSORS.items() for m in missing[s]
                    if abs(y - m) <= spec['event_window']]
            if gaps:
                blocking.append(y)

        n_miss = sum(len(v) for v in missing.values())
        pct = 100 * (1 - n_miss / N_YEARS)
        flag = '  <-- PRIORITY' if blocking else ''
        print(f'{area:24s} coverage {pct:5.1f}%  events {len(ev_years):2d} '
              f'(+{n_pre} before {first})  blocked {len(blocking):2d}{flag}')
        for s in SENSORS:
            if missing[s]:
                print(f'{"":26s}{s:9s} missing : {missing[s]}')
            if partial[s]:
                print(f'{"":26s}{s:9s} partial : {partial[s]} (some of {", ".join(REQUIRED)} absent)')

        if n_miss:
            plan.append({
                'area': area,
                'region': analysis_region(area),
                'bbox': bb,
                'epsg': cfg['epsg'],
                'months': cfg['dry_season_months'],
                'max_cloud_cover': cfg['max_cloud_cover'],
                'missing_landsat': missing['landsat'],
                'missing_sentinel2': missing['sentinel2'],
                'partial_landsat': partial['landsat'],
                'partial_sentinel2': partial['sentinel2'],
                'coverage_pct': round(pct, 1),
                'events_in_bbox': len(ev_years),
                'events_before_archive': n_pre,
                'events_blocked_by_gap': len(blocking),
            })

    plan.sort(key=lambda r: (-r['events_blocked_by_gap'], r['coverage_pct']))

    print('\n=== prioritised downloads ===\n')
    tot_ls = tot_s2 = 0
    for r in plan:
        tot_ls += len(r['missing_landsat'])
        tot_s2 += len(r['missing_sentinel2'])
        print(f"{r['area']:24s} {len(r['missing_landsat']):2d} Landsat years + "
              f"{len(r['missing_sentinel2']):2d} S2 years   "
              f"unblocks {r['events_blocked_by_gap']} events")
    print(f'\ntotal: {tot_ls} Landsat area-years + {tot_s2} Sentinel-2 area-years')

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(plan, indent=2))
    print(f'\nplan saved to {OUT}')
    print('consume with scripts/01_download_landsat.py and scripts/02_download_s2.py '
          'iterating over the plan entries')


if __name__ == '__main__':
    main()
