import sys
import argparse
import subprocess
import zipfile
from pathlib import Path

import pandas as pd
import geopandas as gpd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

BASE_URL = ('https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0770_rgi_v7/'
            'regional_files/RGI2000-v7.0-G')
REGIONS = {
    '16': 'RGI2000-v7.0-G-16_low_latitudes.zip',
    '17': 'RGI2000-v7.0-G-17_southern_andes.zip',
}
ANDES_BBOX = (-82.0, -56.0, -60.0, 13.0)
KEEP = ['rgi_id', 'o1region', 'o2region', 'src_date', 'area_km2', 'zmin_m', 'zmax_m',
        'zmed_m', 'term_type', 'geometry']

RAW_DIR = PROJECT_ROOT / 'data' / 'raw' / 'rgi7'
OUT = PROJECT_ROOT / 'data' / 'raw' / 'rgi7_andes.gpkg'


def fetch(name: str, force: bool) -> Path:
    dst = RAW_DIR / name
    if dst.exists() and zipfile.is_zipfile(dst) and not force:
        return dst
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cookies = RAW_DIR / '.urs_cookies'
    cmd = ['curl', '-f', '-n', '-L', '-c', str(cookies), '-b', str(cookies),
           '-o', str(dst), f'{BASE_URL}/{name}']
    print(f'  downloading {name}')
    if subprocess.run(cmd).returncode != 0 or not zipfile.is_zipfile(dst):
        dst.unlink(missing_ok=True)
        sys.exit('download failed: put NASA Earthdata credentials in ~/.netrc '
                 '(machine urs.earthdata.nasa.gov login <user> password <pass>)')
    return dst


def read_region(zip_path: Path) -> gpd.GeoDataFrame:
    with zipfile.ZipFile(zip_path) as z:
        shp = [n for n in z.namelist() if n.endswith('.shp')]
    g = gpd.read_file(f'zip://{zip_path}!{shp[0]}', bbox=ANDES_BBOX)
    return g[[c for c in KEEP if c in g.columns]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    parts = [read_region(fetch(name, args.force)) for name in REGIONS.values()]
    g = gpd.GeoDataFrame(pd.concat(parts, ignore_index=True), crs=parts[0].crs).to_crs(4326)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g['geometry'] = g.geometry.make_valid()
    OUT.unlink(missing_ok=True)
    g.to_file(OUT, driver='GPKG', layer='rgi7_andes')
    print(f'  glaciers: {len(g)}  area: {g["area_km2"].sum():.0f} km2  -> {OUT.relative_to(PROJECT_ROOT)}')


if __name__ == '__main__':
    main()
