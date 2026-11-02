import os

os.environ.setdefault('GDAL_DISABLE_READDIR_ON_OPEN', 'EMPTY_DIR')
os.environ.setdefault('CPL_VSIL_CURL_ALLOWED_EXTENSIONS', '.tif,.tiff,.jp2')
os.environ.setdefault('GDAL_HTTP_MULTIPLEX', 'YES')
os.environ.setdefault('GDAL_HTTP_VERSION', '2')
os.environ.setdefault('GDAL_HTTP_MAX_RETRY', '5')
os.environ.setdefault('GDAL_HTTP_TIMEOUT', '180')
os.environ.setdefault('GDAL_HTTP_CONNECTTIMEOUT', '30')
os.environ.setdefault('GDAL_HTTP_LOW_SPEED_TIME', '60')
os.environ.setdefault('GDAL_HTTP_LOW_SPEED_LIMIT', '1024')
os.environ.setdefault('GDAL_HTTP_RETRY_DELAY', '2')
os.environ.setdefault('VSI_CACHE', 'TRUE')
os.environ.setdefault('VSI_CACHE_SIZE', '536870912')
os.environ.setdefault('GDAL_CACHEMAX', '1024')

import sys
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import EXPANDED_STUDY_AREAS, active_areas
from src.detection import season_range

LANDSAT_YEARS = list(range(1985, 2017))
MAX_CLOUD = 90
SCENES_PER_TILE = 6
TARGET_RES_M = 30.0
MIN_COVERAGE = 0.98
REDO_COVERAGE = 0.85
MOSAIC_VERSION = 3
PREFETCH = 6

PLATFORMS = ['landsat-4', 'landsat-5', 'landsat-7', 'landsat-8', 'landsat-9']
GREEN_BAND = 'green'
NIR_BAND = 'nir08'
SWIR_BAND = 'swir16'
QA_BAND = 'qa_pixel'
QA_INVALID_BITS = (0, 1, 2, 3, 4)

LS_SCALE = 0.0000275
LS_OFFSET = -0.2


def get_pc_client():
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open(
        'https://planetarycomputer.microsoft.com/api/stac/v1',
        modifier=planetary_computer.sign_inplace,
    )


def target_grid(bbox, epsg):
    dst_crs = f'EPSG:{epsg}'
    left, bottom, right, top = transform_bounds('EPSG:4326', dst_crs, *bbox)
    width = max(1, int(np.ceil((right - left) / TARGET_RES_M)))
    height = max(1, int(np.ceil((top - bottom) / TARGET_RES_M)))
    transform = from_origin(left, top, TARGET_RES_M, TARGET_RES_M)
    return dst_crs, transform, width, height


def read_band_window(item, band, bbox, max_retries=3):
    import planetary_computer

    if band not in item.assets:
        return None, None, None

    href = item.assets[band].href.split('?')[0]
    west, south, east, north = bbox

    for attempt in range(1, max_retries + 1):
        try:
            with rasterio.open(planetary_computer.sign(href)) as src:
                l, b, r, t = transform_bounds('EPSG:4326', src.crs,
                                              west, south, east, north)
                win = from_bounds(l, b, r, t, transform=src.transform)
                win = win.intersection(
                    rasterio.windows.Window(0, 0, src.width, src.height))
                if win.width < 1 or win.height < 1:
                    return None, None, None
                data = src.read(1, window=win).astype(np.float32)
                if data.size == 0:
                    return None, None, None
                return data, src.window_transform(win), src.crs
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(attempt * 3)
            else:
                print(f'    [ERROR] {band}: {exc}')
                return None, None, None
    return None, None, None


def normalized_difference(a, b):
    denom = a + b
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(denom > 0, (a - b) / denom, np.nan).astype(np.float32)


def scene_indices(item, bbox):
    names = (GREEN_BAND, NIR_BAND, SWIR_BAND, QA_BAND)
    with ThreadPoolExecutor(max_workers=len(names)) as pool:
        raw = dict(zip(names, pool.map(lambda b: read_band_window(item, b, bbox), names)))
    bands = {}
    transform = crs = None
    for name in names:
        arr, transform, crs = raw[name]
        if arr is None:
            return None, None, None, None, None
        if bands and arr.shape != next(iter(bands.values())).shape:
            return None, None, None, None, None
        bands[name] = arr

    qa = bands[QA_BAND].astype(np.uint16)
    invalid = np.zeros(qa.shape, dtype=bool)
    for bit in QA_INVALID_BITS:
        invalid |= (qa >> bit) & 1 == 1

    refl = {}
    for name in (GREEN_BAND, NIR_BAND, SWIR_BAND):
        r = bands[name] * LS_SCALE + LS_OFFSET
        invalid |= r <= -0.19
        refl[name] = np.clip(r, 0.0, None)

    mndwi = normalized_difference(refl[GREEN_BAND], refl[SWIR_BAND])
    ndwi = normalized_difference(refl[GREEN_BAND], refl[NIR_BAND])
    mndwi[invalid] = np.nan
    ndwi[invalid] = np.nan
    nir = refl[NIR_BAND].astype(np.float32)
    nir[invalid] = np.nan
    return mndwi, ndwi, nir, transform, crs


def write_nir(path, nir, profile):
    p = dict(profile, dtype='uint16', nodata=0)
    scaled = np.where(np.isfinite(nir), np.clip(np.rint(nir * 10000), 1, 65535), 0)
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(scaled.astype(np.uint16), 1)


def select_scenes(items):
    by_tile = {}
    for it in items:
        p = it.properties
        key = (p.get('landsat:wrs_path'), p.get('landsat:wrs_row'))
        by_tile.setdefault(key, []).append(it)
    groups = []
    for group in by_tile.values():
        group.sort(key=lambda x: x.properties.get('eo:cloud_cover', 100.0))
        groups.append(group[:SCENES_PER_TILE])
    groups.sort(key=lambda g: g[0].properties.get('eo:cloud_cover', 100.0))
    out = []
    for i in range(max(len(g) for g in groups)):
        for group in groups:
            if i < len(group):
                out.append(group[i])
    return out, len(groups)


def build_mosaic(items, bbox, epsg):
    dst_crs, dst_transform, width, height = target_grid(bbox, epsg)
    mosaic = np.full((height, width), np.nan, dtype=np.float32)
    ndwi_m = np.full((height, width), np.nan, dtype=np.float32)
    nir_m = np.full((height, width), np.nan, dtype=np.float32)
    used = []

    with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        for start in range(0, len(items), PREFETCH):
            if np.isfinite(mosaic).mean() >= MIN_COVERAGE:
                break
            batch = items[start:start + PREFETCH]
            for it, (mndwi, ndwi, nir, src_transform, src_crs) in pool.map(
                    lambda x: (x, scene_indices(x, bbox)), batch):
                if mndwi is None:
                    continue
                patch = np.full((height, width), np.nan, dtype=np.float32)
                patch_n = np.full((height, width), np.nan, dtype=np.float32)
                patch_r = np.full((height, width), np.nan, dtype=np.float32)
                for src, dst in ((mndwi, patch), (ndwi, patch_n), (nir, patch_r)):
                    reproject(
                        source=src, destination=dst,
                        src_transform=src_transform, src_crs=src_crs,
                        dst_transform=dst_transform, dst_crs=dst_crs,
                        src_nodata=np.nan, dst_nodata=np.nan,
                        resampling=Resampling.bilinear,
                    )
                fill = np.isnan(mosaic) & np.isfinite(patch) & np.isfinite(patch_n)
                if not fill.any():
                    continue
                mosaic[fill] = patch[fill]
                ndwi_m[fill] = patch_n[fill]
                nir_m[fill] = patch_r[fill]
                used.append({
                    'scene_id': it.id,
                    'platform': it.properties.get('platform'),
                    'date': it.datetime.strftime('%Y-%m-%d'),
                    'cloud_cover': it.properties.get('eo:cloud_cover'),
                    'wrs_path': it.properties.get('landsat:wrs_path'),
                    'wrs_row': it.properties.get('landsat:wrs_row'),
                    'pixels_added': int(fill.sum()),
                })

    coverage = float(np.isfinite(mosaic).mean())
    profile = {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
        'width': width, 'height': height, 'crs': dst_crs,
        'transform': dst_transform, 'nodata': np.nan, 'compress': 'lzw',
    }
    return mosaic, ndwi_m, nir_m, profile, used, coverage


def download_one_area_year(catalog, area_name, cfg, year, out_dir):
    bbox = cfg['bbox']
    date_range = season_range(year, cfg['dry_season_months'])

    mndwi_out = out_dir / 'mndwi.tif'
    ndwi_out = out_dir / 'ndwi.tif'
    nir_out = out_dir / 'nir.tif'
    meta_out = out_dir / 'metadata.json'
    if mndwi_out.exists() and ndwi_out.exists() and nir_out.exists() and meta_out.exists():
        try:
            meta = json.loads(meta_out.read_text())
            if (meta.get('bbox') == bbox
                    and meta.get('version') == MOSAIC_VERSION
                    and meta.get('date_range') == date_range
                    and meta.get('coverage', 0) >= REDO_COVERAGE):
                print(f'  [skip] {area_name}/{year} — mosaic covers current bbox')
                return True
        except Exception:
            pass

    try:
        items = list(catalog.search(
            collections=['landsat-c2-l2'],
            bbox=bbox,
            datetime=date_range,
            query={'eo:cloud_cover': {'lt': MAX_CLOUD},
                   'platform': {'in': PLATFORMS}},
        ).items())
    except Exception as e:
        print(f'  [ERROR] {area_name}/{year} search: {e}')
        return False

    if not items:
        print(f'  [none] {area_name}/{year} — no scenes under {MAX_CLOUD}% cloud')
        return False

    scenes, n_tiles = select_scenes(items)
    print(f'  [dl] {area_name}/{year}  {len(items)} scenes, '
          f'{n_tiles} WRS tiles', end=' ... ', flush=True)

    mosaic, ndwi, nir, profile, used, coverage = build_mosaic(scenes, bbox, cfg['epsg'])

    if not used:
        print('failed')
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(mndwi_out, 'w', **profile) as dst:
        dst.write(mosaic, 1)
    with rasterio.open(ndwi_out, 'w', **profile) as dst:
        dst.write(ndwi, 1)
    write_nir(nir_out, nir, profile)

    meta_out.write_text(json.dumps({
        'version': MOSAIC_VERSION,
        'bbox': bbox,
        'date_range': date_range,
        'epsg': cfg['epsg'],
        'coverage': round(coverage, 4),
        'n_scenes': len(used),
        'scenes': used,
    }, indent=2))

    print(f'ok  {len(used)} scenes, coverage {100*coverage:.1f}%, '
          f'{mndwi_out.stat().st_size/1024:.0f} KB')
    return True


def download_one(area_name, years):
    if area_name not in EXPANDED_STUDY_AREAS:
        print(f'[ERROR] Unknown area: {area_name}')
        return False

    cfg = EXPANDED_STUDY_AREAS[area_name]
    print(f'\n{"="*60}')
    print(f'Landsat mosaic: {area_name}  years {years[0]}-{years[-1]}')
    print(f'  bbox   : {cfg["bbox"]}')
    print(f'  months : {cfg["dry_season_months"]}')

    try:
        catalog = get_pc_client()
    except Exception as e:
        print(f'[ERROR] Cannot connect to Planetary Computer: {e}')
        return False

    success = 0
    for year in years:
        out_dir = PROJECT_ROOT / 'data' / 'raw' / 'landsat' / area_name / str(year)
        if download_one_area_year(catalog, area_name, cfg, year, out_dir):
            success += 1

    print(f'[done] {area_name}: {success}/{len(years)} years')
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download and mosaic Landsat C2 L2 MNDWI/NDWI (1985-2016)')
    parser.add_argument('area_name', nargs='?')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--years', default='1985-2016')
    args = parser.parse_args()

    s, e = args.years.split('-')
    years = list(range(int(s), int(e) + 1))

    if not args.all and not args.area_name:
        parser.print_help()
        sys.exit(1)

    if args.all:
        areas = sorted(active_areas())
        print(f'Landsat mosaic for ALL {len(areas)} areas, {years[0]}-{years[-1]}')
        failed = []
        for name in areas:
            if not download_one(name, years):
                failed.append(name)
        print(f'\n=== SUMMARY ===')
        print(f'Completed: {len(areas) - len(failed)}/{len(areas)}')
        if failed:
            print(f'Failed   : {failed}')
    else:
        sys.exit(0 if download_one(args.area_name, years) else 1)


if __name__ == '__main__':
    main()
