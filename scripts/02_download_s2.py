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
from rasterio.windows import from_bounds, Window
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.transform import from_origin
from affine import Affine

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import (EXPANDED_STUDY_AREAS, EXCLUDED_AREAS,
                                         active_areas)
from src.detection import effective_bbox, season_range

S2_YEARS = list(range(2017, 2026))
TARGET_RES_M = 20.0
MIN_COVERAGE = 0.98
YEAR_RETRIES = 5
RETRY_WAIT_S = 120
REDO_COVERAGE = 0.85
MAX_SCENES = 12
PREFETCH = 4
S2_SCALE = 10000.0
BOA_OFFSET = 1000.0

GREEN_BAND = 'B03'
NIR_BAND = 'B08'
SWIR_BAND = 'B11'
SCL_BAND = 'SCL'

SCL_INVALID = (0, 1, 3, 8, 9, 10)
SCL_WATER = 6
SCENE_CLOUD_MAX = 90


def get_pc_client():
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open(
        'https://planetarycomputer.microsoft.com/api/stac/v1',
        modifier=planetary_computer.sign_inplace,
    )


def parse_year_range(s):
    parts = s.split('-')
    if len(parts) == 2:
        return list(range(int(parts[0]), int(parts[1]) + 1))
    return [int(p) for p in parts]


def target_grid(bbox, epsg):
    dst_crs = f'EPSG:{epsg}'
    left, bottom, right, top = transform_bounds('EPSG:4326', dst_crs, *bbox)
    width = max(1, int(np.ceil((right - left) / TARGET_RES_M)))
    height = max(1, int(np.ceil((top - bottom) / TARGET_RES_M)))
    return dst_crs, from_origin(left, top, TARGET_RES_M, TARGET_RES_M), width, height


def boa_offset(item):
    baseline = str(item.properties.get('s2:processing_baseline', '0'))
    try:
        return BOA_OFFSET if float(baseline) >= 4.0 else 0.0
    except ValueError:
        return 0.0


def read_band_window(item, band, bbox, resampling=Resampling.average,
                     max_retries=3):
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
                win = win.intersection(Window(0, 0, src.width, src.height))
                if win.width < 1 or win.height < 1:
                    return None, None, None

                scale = abs(src.transform.a) / TARGET_RES_M
                if scale < 1.0:
                    out_h = max(1, int(round(win.height * scale)))
                    out_w = max(1, int(round(win.width * scale)))
                    data = src.read(1, window=win, out_shape=(out_h, out_w),
                                    resampling=resampling).astype(np.float32)
                    tr = src.window_transform(win) * Affine.scale(
                        win.width / out_w, win.height / out_h)
                else:
                    data = src.read(1, window=win).astype(np.float32)
                    tr = src.window_transform(win)

                if data.size == 0:
                    return None, None, None
                return data, tr, src.crs
        except Exception as exc:
            if attempt < max_retries:
                time.sleep(attempt * 3)
            else:
                print(f'    [ERROR] {band}: {exc}')
                return None, None, None
    return None, None, None


def scene_indices(item, bbox):
    jobs = {
        SWIR_BAND: Resampling.average,
        GREEN_BAND: Resampling.average,
        NIR_BAND: Resampling.average,
        SCL_BAND: Resampling.nearest,
    }
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {band: pool.submit(read_band_window, item, band, bbox, rs)
                   for band, rs in jobs.items()}
        raw = {band: f.result() for band, f in futures.items()}

    swir, transform, crs = raw[SWIR_BAND]
    if swir is None:
        return None, None, None, None, None, None

    shape_hw = swir.shape
    off = boa_offset(item)

    def to_grid(band):
        arr, src_transform, src_crs = raw[band]
        if arr is None:
            return None
        out = np.full(shape_hw, np.nan, dtype=np.float32)
        reproject(
            source=arr, destination=out,
            src_transform=src_transform, src_crs=src_crs,
            dst_transform=transform, dst_crs=crs,
            src_nodata=0, dst_nodata=np.nan,
            resampling=Resampling.average,
        )
        return out

    green = to_grid(GREEN_BAND)
    nir = to_grid(NIR_BAND)
    if green is None:
        return None, None, None, None, None, None

    water = np.full(shape_hw, np.nan, dtype=np.float32)
    scl, scl_transform, scl_crs = raw[SCL_BAND]
    if scl is not None:
        if scl.shape != shape_hw:
            resampled = np.zeros(shape_hw, dtype=np.float32)
            reproject(
                source=scl, destination=resampled,
                src_transform=scl_transform, src_crs=scl_crs,
                dst_transform=transform, dst_crs=crs,
                resampling=Resampling.nearest,
            )
            scl = resampled
        classes = np.rint(scl).astype(np.int16)
        invalid = np.isin(classes, SCL_INVALID)
        water = np.where(invalid, np.nan, classes == SCL_WATER).astype(np.float32)
        swir = np.where(invalid, np.nan, swir)
        green = np.where(invalid, np.nan, green)
        if nir is not None:
            nir = np.where(invalid, np.nan, nir)

    swir = np.where(swir == 0, np.nan, swir)
    green = np.clip((green - off) / S2_SCALE, 0.0, None)
    swir = np.clip((swir - off) / S2_SCALE, 0.0, None)

    with np.errstate(divide='ignore', invalid='ignore'):
        denom = green + swir
        mndwi = np.where(denom > 0, (green - swir) / denom, np.nan).astype(np.float32)
        if nir is not None:
            nir = np.clip((nir - off) / S2_SCALE, 0.0, None)
            denom2 = green + nir
            ndwi = np.where(denom2 > 0, (green - nir) / denom2, np.nan).astype(np.float32)
        else:
            nir = np.full(shape_hw, np.nan, dtype=np.float32)
            ndwi = np.full(shape_hw, np.nan, dtype=np.float32)

    return mndwi, ndwi, nir.astype(np.float32), water, transform, crs


def write_scl_water(path, water, profile):
    p = dict(profile, dtype='uint8', nodata=255)
    out = np.where(np.isfinite(water), water, 255).astype(np.uint8)
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(out, 1)


def replay_scl_water(catalog, meta, profile):
    collection = catalog.get_collection('sentinel-2-l2a')
    shape_hw = (profile['height'], profile['width'])
    water = np.full(shape_hw, np.nan, dtype=np.float32)
    for sc in meta['scenes']:
        item = collection.get_item(sc['scene_id'])
        if item is None:
            continue
        scl, tr, crs = read_band_window(item, SCL_BAND, meta['download_bbox'],
                                        Resampling.nearest)
        if scl is None:
            continue
        grid = np.zeros(shape_hw, dtype=np.float32)
        reproject(source=scl, destination=grid, src_transform=tr, src_crs=crs,
                  dst_transform=profile['transform'], dst_crs=profile['crs'],
                  resampling=Resampling.nearest)
        classes = np.rint(grid).astype(np.int16)
        fill = np.isnan(water) & ~np.isin(classes, SCL_INVALID)
        water[fill] = classes[fill] == SCL_WATER
    return water


def write_nir(path, nir, profile):
    p = dict(profile, dtype='uint16', nodata=0)
    scaled = np.where(np.isfinite(nir), np.clip(np.rint(nir * 10000), 1, 65535), 0)
    with rasterio.open(path, 'w', **p) as dst:
        dst.write(scaled.astype(np.uint16), 1)


def select_scenes(items):
    by_orbit = {}
    for it in items:
        key = (it.properties.get('sat:relative_orbit'),
               it.properties.get('s2:mgrs_tile'))
        by_orbit.setdefault(key, []).append(it)
    for group in by_orbit.values():
        group.sort(key=lambda x: x.properties.get('eo:cloud_cover', 100.0))

    order = sorted(by_orbit.values(),
                   key=lambda g: g[0].properties.get('eo:cloud_cover', 100.0))
    out = []
    for i in range(max(len(g) for g in order)):
        for group in order:
            if i < len(group):
                out.append(group[i])
    return out


def build_mosaic(items, bbox, epsg):
    dst_crs, dst_transform, width, height = target_grid(bbox, epsg)
    mndwi_m = np.full((height, width), np.nan, dtype=np.float32)
    ndwi_m = np.full((height, width), np.nan, dtype=np.float32)
    nir_m = np.full((height, width), np.nan, dtype=np.float32)
    water_m = np.full((height, width), np.nan, dtype=np.float32)
    used = []

    queue = items[:MAX_SCENES]
    with ThreadPoolExecutor(max_workers=PREFETCH) as pool:
        for batch_start in range(0, len(queue), PREFETCH):
            if np.isfinite(mndwi_m).mean() >= MIN_COVERAGE:
                break
            batch = queue[batch_start:batch_start + PREFETCH]
            results = list(pool.map(lambda it: (it, scene_indices(it, bbox)), batch))
            for it, (mndwi, ndwi, nir, water, src_transform, src_crs) in results:
                if mndwi is None:
                    continue
                patch_m = np.full((height, width), np.nan, dtype=np.float32)
                patch_n = np.full((height, width), np.nan, dtype=np.float32)
                patch_r = np.full((height, width), np.nan, dtype=np.float32)
                patch_w = np.full((height, width), np.nan, dtype=np.float32)
                for src, dst, rs in ((mndwi, patch_m, Resampling.bilinear),
                                     (ndwi, patch_n, Resampling.bilinear),
                                     (nir, patch_r, Resampling.bilinear),
                                     (water, patch_w, Resampling.nearest)):
                    reproject(
                        source=src, destination=dst,
                        src_transform=src_transform, src_crs=src_crs,
                        dst_transform=dst_transform, dst_crs=dst_crs,
                        src_nodata=np.nan, dst_nodata=np.nan,
                        resampling=rs,
                    )
                fill = np.isnan(mndwi_m) & np.isfinite(patch_m) & np.isfinite(patch_n)
                if not fill.any():
                    continue
                mndwi_m[fill] = patch_m[fill]
                ndwi_m[fill] = patch_n[fill]
                nir_m[fill] = patch_r[fill]
                water_m[fill] = patch_w[fill]
                used.append({
                    'scene_id': it.id,
                    'platform': it.properties.get('platform'),
                    'mgrs_tile': it.properties.get('s2:mgrs_tile'),
                    'date': it.datetime.strftime('%Y-%m-%d'),
                    'cloud_cover': it.properties.get('eo:cloud_cover'),
                    'processing_baseline': it.properties.get('s2:processing_baseline'),
                    'pixels_added': int(fill.sum()),
                })

    coverage = float(np.isfinite(mndwi_m).mean())
    profile = {
        'driver': 'GTiff', 'dtype': 'float32', 'count': 1,
        'width': width, 'height': height, 'crs': dst_crs,
        'transform': dst_transform, 'nodata': np.nan, 'compress': 'lzw',
    }
    return mndwi_m, ndwi_m, nir_m, water_m, profile, used, coverage


def download_one_area_year(catalog, area_name, cfg, year, out_dir, dl_bbox=None):
    bbox = cfg['bbox']
    dl_bbox = dl_bbox or bbox
    months = cfg['dry_season_months']
    date_range = season_range(year, months)

    mndwi_out = out_dir / 'mndwi.tif'
    ndwi_out = out_dir / 'ndwi.tif'
    nir_out = out_dir / 'nir.tif'
    water_out = out_dir / 'scl_water.tif'
    meta_out = out_dir / 'metadata.json'
    if mndwi_out.exists() and ndwi_out.exists() and nir_out.exists() and meta_out.exists():
        try:
            meta = json.loads(meta_out.read_text())
            built_range = meta.get('date_range') or season_range(year, sorted(months))
            current = (meta.get('bbox') == bbox and built_range == date_range
                       and meta.get('coverage', 0) >= REDO_COVERAGE)
        except Exception:
            current = False
        if current and water_out.exists():
            print(f'  [skip] {area_name}/{year} — mosaic covers current bbox')
            return True
        if current:
            print(f'  [scl] {area_name}/{year}  {meta["n_scenes"]} scenes', end=' ... ', flush=True)
            with rasterio.open(mndwi_out) as src:
                profile = src.profile
            water = replay_scl_water(catalog, meta, profile)
            write_scl_water(water_out, water, profile)
            print(f'ok  water {int(np.nansum(water))} px')
            return True

    try:
        items = list(catalog.search(
            collections=['sentinel-2-l2a'],
            bbox=dl_bbox,
            datetime=date_range,
            query={'eo:cloud_cover': {'lt': SCENE_CLOUD_MAX}},
        ).items())
    except Exception as e:
        print(f'  [ERROR] {area_name}/{year} search: {e}')
        return False

    if not items:
        print(f'  [none] {area_name}/{year} — no scenes under '
              f'{SCENE_CLOUD_MAX}% cloud')
        return False

    scenes = select_scenes(items)
    print(f'  [dl] {area_name}/{year}  {len(items)} scenes, '
          f'{len(scenes)} MGRS tiles', end=' ... ', flush=True)

    mndwi, ndwi, nir, water, profile, used, coverage = build_mosaic(scenes, dl_bbox, cfg['epsg'])

    if not used:
        print('failed')
        return False

    out_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(mndwi_out, 'w', **profile) as dst:
        dst.write(mndwi, 1)
    with rasterio.open(ndwi_out, 'w', **profile) as dst:
        dst.write(ndwi, 1)
    write_nir(nir_out, nir, profile)
    write_scl_water(water_out, water, profile)

    meta_out.write_text(json.dumps({
        'bbox': bbox,
        'download_bbox': dl_bbox,
        'date_range': date_range,
        'epsg': cfg['epsg'],
        'resolution_m': TARGET_RES_M,
        'coverage': round(coverage, 4),
        'n_scenes': len(used),
        'scenes': used,
    }, indent=2))

    size_mb = (mndwi_out.stat().st_size + ndwi_out.stat().st_size) / 1024 / 1024
    print(f'ok  {len(used)} tiles, coverage {100*coverage:.1f}%, {size_mb:.0f} MB')
    return True


def download_one(area_name, years):
    if area_name not in EXPANDED_STUDY_AREAS:
        print(f'[ERROR] Unknown area: {area_name}')
        return False
    if area_name in EXCLUDED_AREAS:
        print(f'[excluded] {area_name} — {EXCLUDED_AREAS[area_name]}')
        return True

    cfg = EXPANDED_STUDY_AREAS[area_name]
    print(f'\n{"="*60}')
    print(f'Sentinel-2 mosaic: {area_name}  years {years[0]}-{years[-1]}')
    print(f'  bbox   : {cfg["bbox"]}')
    print(f'  months : {cfg["dry_season_months"]}')
    print(f'  cloud  : {cfg["max_cloud_cover"]}%')

    try:
        catalog = get_pc_client()
    except Exception as e:
        print(f'[ERROR] Cannot connect to Planetary Computer: {e}')
        return False

    dl_bbox = effective_bbox(area_name, cfg['bbox'], cfg['epsg'])
    if dl_bbox != list(cfg['bbox']):
        print(f'  clipped to ice buffer: {[round(v, 3) for v in dl_bbox]}')

    success = 0
    for year in years:
        out_dir = PROJECT_ROOT / 'data' / 'raw' / 'sentinel2' / area_name / str(year)
        for attempt in range(YEAR_RETRIES):
            try:
                if download_one_area_year(catalog, area_name, cfg, year, out_dir, dl_bbox):
                    success += 1
                break
            except Exception as e:
                print(f'failed ({type(e).__name__}); retry {attempt + 1}/{YEAR_RETRIES} in {RETRY_WAIT_S}s')
                time.sleep(RETRY_WAIT_S)
                catalog = get_pc_client()

    print(f'[done] {area_name}: {success}/{len(years)} years')
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download and mosaic Sentinel-2 L2A MNDWI/NDWI (2017-2025)')
    parser.add_argument('area_name', nargs='?')
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--years', default=None)
    args = parser.parse_args()

    years = parse_year_range(args.years) if args.years else S2_YEARS

    if not args.all and not args.area_name:
        parser.print_help()
        sys.exit(1)

    if args.all:
        areas = sorted(active_areas())
        print(f'Sentinel-2 mosaic for {len(areas)} active areas, {years[0]}-{years[-1]}')
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
