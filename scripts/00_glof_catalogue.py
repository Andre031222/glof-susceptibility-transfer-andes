import sys
import argparse
import hashlib
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

ZENODO_RECORD = '23370149'
DB_VERSION = '4.2'
DB_FILE = 'glofdatabase_v4-2.ods'
DB_DOI = '10.5281/zenodo.7330344'
SHEETS = ['RGI60_16_LowLatitudes', 'RGI60_17_SouthAndes']

RAW_DIR = PROJECT_ROOT / 'data' / 'raw' / 'glofdb'
OUT_DIR = PROJECT_ROOT / 'data' / 'source'
OUT_GPKG = OUT_DIR / 'historical_glofs.gpkg'
OUT_CSV = OUT_DIR / 'historical_glofs.csv'


def download(force: bool) -> Path:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    dst = RAW_DIR / DB_FILE
    if dst.exists() and not force:
        return dst
    url = f'https://zenodo.org/api/records/{ZENODO_RECORD}/files/{DB_FILE}/content'
    print(f'  downloading {url}')
    urllib.request.urlretrieve(url, dst)
    return dst


def first_year(*cols: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=cols[0].index)
    for c in cols:
        y = pd.to_numeric(c.astype(str).str.extract(r'((?:1[6-9]|20)\d\d)')[0], errors='coerce')
        out = out.fillna(y)
    return out


def dam_class(t) -> str:
    t = str(t).lower()
    if t.startswith('ice') or t in ('supraglacial', 'subglacial'):
        return 'ice'
    if 'moraine' in t:
        return 'moraine'
    if 'bedrock' in t:
        return 'bedrock'
    return 'unknown'


def clean_text(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace('﻿', '', regex=False).str.strip().replace({'nan': ''})


def build(path: Path) -> gpd.GeoDataFrame:
    sheets = pd.read_excel(path, engine='odf', sheet_name=SHEETS)
    g = pd.concat([d.assign(rgi_region=k) for k, d in sheets.items()], ignore_index=True)
    g['lon'] = pd.to_numeric(g['Longitude'], errors='coerce')
    g['lat'] = pd.to_numeric(g['Latitude'], errors='coerce')
    g = g[g['lon'].between(-82, -60) & g['lat'].between(-56, 2)].copy()

    g['year'] = first_year(g['Date'], g['Date_Min'], g['Date_Max'])
    g = g[g['year'].notna()].copy()
    g['year'] = g['year'].astype(int)

    name = clean_text(g['Lake'])
    name = name.where(name != '', clean_text(g['Glacier']))
    g['lake'] = name.where(name != '', 'unnamed')
    g['country'] = clean_text(g['Country'])
    g['lake_type'] = g['Lake_type'].map(dam_class)
    g['dam_type_raw'] = clean_text(g['Lake_type'])
    g['trigger'] = clean_text(g['Mechanism']).replace('', 'unknown')
    g['date'] = clean_text(g['Date'])
    g['reference'] = clean_text(g['Reference'])
    g['rgi_glacier_id'] = clean_text(g['RGI_Glacier_Id'])

    key = g['lon'].round(4).astype(str) + '_' + g['lat'].round(4).astype(str)
    g['source_lake_key'] = key.map(lambda k: hashlib.md5(k.encode()).hexdigest()[:10])
    g = g.sort_values(['source_lake_key', 'year']).reset_index(drop=True)
    g['event_id'] = [f'GDB{DB_VERSION.replace(".", "")}_{i:04d}' for i in range(len(g))]
    g['is_repeat'] = g.duplicated('source_lake_key', keep='first')
    g['database'] = f'GLOF database v{DB_VERSION} (doi:{DB_DOI})'

    cols = ['event_id', 'lake', 'country', 'year', 'date', 'lake_type', 'dam_type_raw',
            'trigger', 'source_lake_key', 'is_repeat', 'rgi_glacier_id', 'rgi_region',
            'reference', 'database', 'lon', 'lat']
    return gpd.GeoDataFrame(g[cols], geometry=gpd.points_from_xy(g['lon'], g['lat']), crs='EPSG:4326')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    path = download(args.force)
    ev = build(path)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ev.to_file(OUT_GPKG, driver='GPKG')
    ev.drop(columns='geometry').to_csv(OUT_CSV, index=False)

    lakes = ev.drop_duplicates('source_lake_key')
    print(f'  events      : {len(ev)}  ({ev["year"].min()}-{ev["year"].max()})')
    print(f'  source lakes: {len(lakes)}')
    print(ev.groupby(['country', 'lake_type']).size().to_string())
    print(f'  -> {OUT_GPKG.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
