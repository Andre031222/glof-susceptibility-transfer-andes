import sys
import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.warp import reproject

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import (EXPANDED_STUDY_AREAS, EXCLUDED_AREAS,
                                         active_areas)
from src.detection import (detect_lakes, MIN_AREA_M2, MNDWI_THRESH, NDWI_THRESH, NIR_MAX,
                           GLACIER_BUFFER_M)

SENSORS = {
    'sentinel2': {'raw': 'sentinel2', 'out': 'lakes_s2'},
    'landsat': {'raw': 'landsat', 'out': 'lakes_landsat'},
}
COMMON_RES_M = 30.0

OUT_COLUMNS = ['geometry', 'lake_id', 'area_name', 'year', 'area_m2',
               'source', 'satellite', 'scene_date']


def read_raster(path, res_m=COMMON_RES_M, scale=None):
    if not path.exists():
        return None, None, None
    with rasterio.open(path) as src:
        arr = src.read(1).astype(np.float32)
        if src.nodata is not None and not np.isnan(src.nodata):
            arr = np.where(arr == src.nodata, np.nan, arr)
        transform, crs = src.transform, src.crs
    if scale is not None:
        arr = arr * np.float32(scale)
    if abs(abs(transform.a) - res_m) < 1e-6:
        return arr, transform, crs
    w = int(np.ceil(arr.shape[1] * abs(transform.a) / res_m))
    h = int(np.ceil(arr.shape[0] * abs(transform.e) / res_m))
    dst_transform = from_origin(transform.c, transform.f, res_m, res_m)
    out = np.full((h, w), np.nan, dtype=np.float32)
    reproject(source=arr, destination=out, src_transform=transform, src_crs=crs,
              dst_transform=dst_transform, dst_crs=crs, src_nodata=np.nan,
              dst_nodata=np.nan, resampling=Resampling.average)
    return out, dst_transform, crs


def read_metadata(year_dir, year):
    satellite, scene_date = 'unknown', f'{year}-01-01'
    meta_file = year_dir / 'metadata.json'
    if meta_file.exists():
        try:
            scenes = json.loads(meta_file.read_text()).get('scenes') or []
            if scenes:
                main = max(scenes, key=lambda s: s.get('pixels_added', 0))
                scene_date = main.get('date', scene_date)
                platforms = sorted({s.get('platform') for s in scenes if s.get('platform')})
                if platforms:
                    satellite = '+'.join(platforms)
        except Exception:
            pass
    return satellite, scene_date


def detect_area_sensor(area_name, sensor, force=False):
    cfg = EXPANDED_STUDY_AREAS[area_name]
    spec = SENSORS[sensor]

    raw_dir = PROJECT_ROOT / 'data' / 'raw' / spec['raw'] / area_name
    out_dir = PROJECT_ROOT / 'data' / 'processed' / spec['out']
    out_dir.mkdir(parents=True, exist_ok=True)

    if not raw_dir.exists():
        print(f'  [skip] {area_name} / {sensor} — no raw data')
        return 0

    years = sorted(int(d.name) for d in raw_dir.iterdir()
                   if d.is_dir() and d.name.isdigit())
    if not years:
        return 0

    dem_path = PROJECT_ROOT / 'data' / 'interim' / 'dem' / f'{area_name}_dem_utm.tif'
    slope_path = PROJECT_ROOT / 'data' / 'interim' / 'terrain' / area_name / 'slope.tif'
    total = 0

    for year in years:
        out_file = out_dir / f'{area_name}_{year}.gpkg'
        if out_file.exists() and not force:
            try:
                existing = gpd.read_file(out_file)
                print(f'  [skip] {sensor} {year} — {len(existing)} lakes')
                total += len(existing)
                continue
            except Exception:
                out_file.unlink()

        year_dir = raw_dir / str(year)
        mndwi, transform, crs = read_raster(year_dir / 'mndwi.tif')
        if mndwi is None:
            print(f'  [skip] {sensor} {year} — no mndwi.tif')
            continue

        ndwi, _, _ = read_raster(year_dir / 'ndwi.tif')
        nir, _, _ = read_raster(year_dir / 'nir.tif', scale=1e-4)
        if ndwi is None or nir is None:
            print(f'  [skip] {sensor} {year} — ndwi.tif or nir.tif missing (re-run the download)')
            continue
        scl_water = None
        if sensor == 'sentinel2':
            scl_water, _, _ = read_raster(year_dir / 'scl_water.tif')
            if scl_water is None:
                print(f'  [skip] {sensor} {year} — scl_water.tif missing (re-run 02_download_s2.py)')
                continue

        satellite, scene_date = read_metadata(year_dir, year)
        gdf, st = detect_lakes(mndwi, transform, crs, area_name, cfg, year,
                               sensor, scene_date, ndwi=ndwi, nir=nir,
                               dem_path=dem_path, slope_path=slope_path,
                               satellite=satellite, scl_water=scl_water)

        print(f'  {sensor:9s} {year}  water={st["water_px"]:>8d} '
              f'elev={st["after_elev"]:>8d} slope={st["after_slope"]:>8d} poly={st["polygons"]:>5d} '
              f'area={st["after_area"]:>5d} glacier={st["after_glacier"]:>5d}')

        if gdf.empty:
            gpd.GeoDataFrame(columns=OUT_COLUMNS, geometry='geometry',
                             crs=crs).to_file(out_file, driver='GPKG')
        else:
            gdf.to_file(out_file, driver='GPKG')
            total += len(gdf)

    return total


def detect_one_area(area_name, sensors, force=False):
    if area_name not in EXPANDED_STUDY_AREAS:
        print(f'[ERROR] Unknown area: {area_name}')
        return False
    if area_name in EXCLUDED_AREAS:
        print(f'[excluded] {area_name} — {EXCLUDED_AREAS[area_name]}')
        return True

    cfg = EXPANDED_STUDY_AREAS[area_name]
    print(f'\n{"="*70}')
    print(f'{area_name}   elev [{cfg["elev_min_m"]},{cfg["elev_max_m"]}]m   '
          f'sensors: {", ".join(sensors)}')

    total = sum(detect_area_sensor(area_name, s, force) for s in sensors)
    print(f'[done] {area_name}: {total} lakes')
    return True


def main():
    parser = argparse.ArgumentParser(
        description=f'Detect glacial lakes from sensor mosaics under identical '
                    f'criteria (MNDWI > {MNDWI_THRESH}, NDWI > {NDWI_THRESH}, NIR < {NIR_MAX}, '
                    f'>= {MIN_AREA_M2:.0f} m2, '
                    f'within {GLACIER_BUFFER_M:.0f} m of RGI 7.0)')
    parser.add_argument('area_name', nargs='?')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--sensor', choices=['sentinel2', 'landsat', 'all'],
                        default='all')
    parser.add_argument('--force', action='store_true',
                        help='recompute even if the output gpkg exists')
    args = parser.parse_args()

    if not args.all and not args.area_name:
        parser.print_help()
        sys.exit(1)

    warnings.filterwarnings('ignore', category=UserWarning)
    sensors = list(SENSORS) if args.sensor == 'all' else [args.sensor]

    if args.all:
        areas = sorted(active_areas())
        for name, reason in EXCLUDED_AREAS.items():
            print(f'[excluded] {name} — {reason}')
        print(f'Lake detection for {len(areas)} active areas')
        failed = []
        for name in areas:
            try:
                if not detect_one_area(name, sensors, args.force):
                    failed.append(name)
            except Exception as e:
                print(f'[ERROR] {name}: {e}')
                failed.append(name)
        print(f'\n=== SUMMARY ===')
        print(f'Completed: {len(areas) - len(failed)}/{len(areas)}')
        if failed:
            print(f'Failed   : {failed}')
    else:
        sys.exit(0 if detect_one_area(args.area_name, sensors, args.force) else 1)


if __name__ == '__main__':
    main()
