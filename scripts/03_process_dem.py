import sys
import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds as warp_bounds
from scipy.ndimage import (convolve, maximum_filter,
                           minimum_filter, uniform_filter, binary_dilation,
                           distance_transform_edt)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import (EXPANDED_STUDY_AREAS, EXCLUDED_AREAS,
                                         active_areas)

EDGE_GUARD_PX = 2
FLAT_SLOPE_DEG = 2.0
TERRAIN_VERSION = 2
NASADEM_NODATA = -32768


def fill_nodata_nearest(dem, nan_mask):
    if not nan_mask.any():
        return dem.copy()
    if nan_mask.all():
        return np.zeros_like(dem)
    idx = distance_transform_edt(nan_mask, return_distances=False,
                                 return_indices=True)
    return dem[tuple(idx)]


def get_pc_client():
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open(
        "https://planetarycomputer.microsoft.com/api/stac/v1",
        modifier=planetary_computer.sign_inplace,
    )


def tile_covers(path: Path, item_bbox, bbox_wgs84, tol=2 / 3600) -> bool:
    need = (max(item_bbox[0], bbox_wgs84[0]), max(item_bbox[1], bbox_wgs84[1]),
            min(item_bbox[2], bbox_wgs84[2]), min(item_bbox[3], bbox_wgs84[3]))
    try:
        with rasterio.open(path) as src:
            have = warp_bounds(src.crs, "EPSG:4326", *src.bounds)
    except Exception:
        return False
    return (have[0] <= need[0] + tol and have[1] <= need[1] + tol
            and have[2] >= need[2] - tol and have[3] >= need[3] - tol)


def download_nasadem(catalog, bbox_wgs84: tuple, raw_dem_dir: Path) -> list:
    import planetary_computer

    items = list(catalog.search(collections=['nasadem'], bbox=bbox_wgs84).items())
    if not items:
        raise RuntimeError(f"No NASADEM tiles for bbox {bbox_wgs84}")

    west, south, east, north = bbox_wgs84
    paths = []
    failed = []
    for item in items:
        out = raw_dem_dir / f"{item.id}_elevation.tif"
        if out.exists() and tile_covers(out, item.bbox, bbox_wgs84):
            print(f"    [ok] {item.id}")
            paths.append(out)
            continue

        url = planetary_computer.sign(item.assets['elevation'].href.split('?')[0])
        for attempt in range(1, 4):
            try:
                with rasterio.open(url) as src:
                    l, b, r, t = warp_bounds("EPSG:4326", src.crs, west, south, east, north)
                    win = from_bounds(l, b, r, t, transform=src.transform)
                    c0 = max(0, int(np.floor(win.col_off)))
                    r0 = max(0, int(np.floor(win.row_off)))
                    c1 = min(src.width, int(np.ceil(win.col_off + win.width)))
                    r1 = min(src.height, int(np.ceil(win.row_off + win.height)))
                    if c1 <= c0 or r1 <= r0:
                        break
                    win = rasterio.windows.Window(c0, r0, c1 - c0, r1 - r0)
                    data = src.read(1, window=win)
                    if data.size == 0:
                        break
                    profile = src.profile | {
                        'compress': 'lzw',
                        'width': data.shape[1],
                        'height': data.shape[0],
                        'transform': src.window_transform(win),
                        'count': 1,
                    }
                with rasterio.open(out, 'w', **profile) as dst:
                    dst.write(data, 1)
                paths.append(out)
                print(f"    [dl] {item.id} ({out.stat().st_size/1e6:.1f} MB)")
                break
            except Exception as exc:
                if out.exists():
                    out.unlink()
                if attempt < 3:
                    time.sleep(attempt * 3)
                else:
                    print(f"    [ERROR] {item.id}: {exc}")
                    failed.append(item.id)
    if failed:
        sys.exit(f"[ERROR] NASADEM: {len(failed)} of {len(items)} tiles could not be read "
                 f"({', '.join(failed)}); refusing to build a DEM with holes")
    return paths


def mosaic_and_reproject(dem_tiles: list, out_path: Path, target_epsg: int):
    srcs = [rasterio.open(p) for p in dem_tiles]
    src_nodata = srcs[0].nodata if srcs[0].nodata is not None else NASADEM_NODATA
    mosaic, mosaic_transform = merge(srcs, nodata=src_nodata)
    src_crs = srcs[0].crs
    for s in srcs:
        s.close()

    data = mosaic[0].astype(np.float32)
    data[mosaic[0] == src_nodata] = np.nan

    dst_crs = rasterio.crs.CRS.from_epsg(target_epsg)
    transform, width, height = calculate_default_transform(
        src_crs, dst_crs,
        mosaic.shape[2], mosaic.shape[1],
        left=mosaic_transform.c,
        bottom=mosaic_transform.f + mosaic_transform.e * mosaic.shape[1],
        right=mosaic_transform.c + mosaic_transform.a * mosaic.shape[2],
        top=mosaic_transform.f,
    )

    out_array = np.full((height, width), np.nan, dtype=np.float32)
    reproject(
        source=data,
        destination=out_array,
        src_transform=mosaic_transform,
        src_crs=src_crs,
        dst_transform=transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
        src_nodata=np.nan,
        dst_nodata=np.nan,
    )

    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'width': width,
        'height': height,
        'count': 1,
        'crs': dst_crs,
        'transform': transform,
        'compress': 'lzw',
        'nodata': np.nan,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(out_array, 1)
    return out_path, profile


def compute_slope_aspect(dem: np.ndarray, res_m: float):
    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / (8 * res_m)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / (8 * res_m)
    dzdx = convolve(dem, kx, mode='nearest')
    dzdy = convolve(dem, ky, mode='nearest')
    slope_deg = np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2))).astype(np.float32)
    aspect_deg = (np.degrees(np.arctan2(dzdx, -dzdy)) % 360).astype(np.float32)
    return slope_deg, aspect_deg


def compute_tri(dem: np.ndarray) -> np.ndarray:
    h, w = dem.shape
    padded = np.pad(dem.astype(np.float64), 1, mode='edge')
    total = np.zeros((h, w), dtype=np.float64)
    for dr in range(3):
        for dc in range(3):
            if dr == 1 and dc == 1:
                continue
            total += np.abs(padded[dr:dr + h, dc:dc + w] - dem)
    return (total / 8.0).astype(np.float32)


def compute_curvature(dem: np.ndarray, res_m: float) -> np.ndarray:
    k_d2x = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]], dtype=np.float32) / res_m**2
    k_d2y = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]], dtype=np.float32) / res_m**2
    return (-(convolve(dem, k_d2x, mode='nearest') + convolve(dem, k_d2y, mode='nearest'))).astype(np.float32)


def compute_roughness(dem: np.ndarray) -> np.ndarray:
    return (maximum_filter(dem, size=3, mode='nearest') - minimum_filter(dem, size=3, mode='nearest')).astype(np.float32)


def compute_vrm(slope_deg: np.ndarray, aspect_deg: np.ndarray) -> np.ndarray:
    slope_rad = np.radians(np.nan_to_num(slope_deg, nan=0.0))
    aspect_rad = np.radians(np.nan_to_num(aspect_deg, nan=0.0))
    x = np.sin(slope_rad) * np.cos(aspect_rad)
    y = np.sin(slope_rad) * np.sin(aspect_rad)
    z = np.cos(slope_rad)
    xsum = uniform_filter(x, size=3, mode='nearest') * 9
    ysum = uniform_filter(y, size=3, mode='nearest') * 9
    zsum = uniform_filter(z, size=3, mode='nearest') * 9
    resultant = np.sqrt(xsum**2 + ysum**2 + zsum**2)
    return np.clip(1.0 - resultant / 9.0, 0, 1).astype(np.float32)


def save_raster(data: np.ndarray, ref_profile: dict, out_path: Path):
    profile = ref_profile.copy()
    profile.update(dtype='float32', count=1, compress='lzw', nodata=np.nan)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(data.astype(np.float32), 1)


def process_one(area_name: str) -> bool:
    if area_name not in EXPANDED_STUDY_AREAS:
        print(f"[ERROR] Unknown area: {area_name}")
        return False

    cfg = EXPANDED_STUDY_AREAS[area_name]
    target_epsg = cfg['epsg']
    bbox = cfg['bbox']

    dem_utm_path = PROJECT_ROOT / 'data' / 'interim' / 'dem' / f"{area_name}_dem_utm.tif"
    terrain_dir = PROJECT_ROOT / 'data' / 'interim' / 'terrain' / area_name
    derivatives = ['slope', 'aspect', 'tri', 'curvature', 'roughness', 'vrm']

    stamp_path = terrain_dir / 'source_bbox.json'
    raw_dem_dir = PROJECT_ROOT / 'data' / 'raw' / 'dem' / area_name
    all_exist = dem_utm_path.exists() and all((terrain_dir / f"{d}.tif").exists()
                                              for d in derivatives)
    if all_exist and stamp_path.exists():
        try:
            stamp = json.loads(stamp_path.read_text())
        except Exception:
            stamp = {}
        if (stamp.get('bbox') == list(bbox) and stamp.get('epsg') == target_epsg
                and stamp.get('terrain_version', 0) == TERRAIN_VERSION):
            print(f"[skip] {area_name} — terrain matches current bbox")
            return True
        print(f"  [stale] {area_name} — terrain built for bbox {stamp.get('bbox')} "
              f"EPSG:{stamp.get('epsg')} v{stamp.get('terrain_version')}, config now "
              f"{list(bbox)} EPSG:{target_epsg} v{TERRAIN_VERSION}; rebuilding")
        if dem_utm_path.exists():
            dem_utm_path.unlink()
        shutil.rmtree(raw_dem_dir, ignore_errors=True)
    elif all_exist:
        print(f"  [stale] {area_name} — terrain has no bbox stamp; rebuilding")
        if dem_utm_path.exists():
            dem_utm_path.unlink()
        shutil.rmtree(raw_dem_dir, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"DEM processing: {area_name}  EPSG:{target_epsg}  bbox:{bbox}")

    raw_dem_dir.mkdir(parents=True, exist_ok=True)

    if not dem_utm_path.exists():
        try:
            catalog = get_pc_client()
        except Exception as e:
            print(f"[ERROR] Planetary Computer: {e}")
            return False

        print(f"  Downloading NASADEM...")
        dem_tiles = download_nasadem(catalog, bbox, raw_dem_dir)
        if not dem_tiles:
            print(f"[ERROR] No DEM tiles for {area_name}")
            return False

        print(f"  Mosaicking -> EPSG:{target_epsg}...")
        dem_utm_path, _ = mosaic_and_reproject(dem_tiles, dem_utm_path, target_epsg)
        print(f"  [saved] {dem_utm_path.name} ({dem_utm_path.stat().st_size/1e6:.1f} MB)")
    else:
        print(f"  [ok] UTM DEM: {dem_utm_path.name}")

    terrain_dir.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dem_utm_path) as src:
        dem = src.read(1).astype(np.float32)
        profile = src.profile.copy()
        res_m = abs(src.transform.a)

    nan_mask = np.isnan(dem)
    dem_filled = fill_nodata_nearest(dem, nan_mask)
    edge_mask = binary_dilation(nan_mask, iterations=EDGE_GUARD_PX)

    print(f"  Computing derivatives (res={res_m:.0f}m, shape={dem.shape}, "
          f"nodata={100*nan_mask.mean():.1f}%)...")

    slope_deg, aspect_deg = compute_slope_aspect(dem_filled, res_m)
    slope_deg[edge_mask] = np.nan
    aspect_deg[edge_mask] = np.nan
    aspect_deg[slope_deg < FLAT_SLOPE_DEG] = np.nan

    tri = compute_tri(dem_filled)
    tri[edge_mask] = np.nan

    curv = compute_curvature(dem_filled, res_m)
    curv[edge_mask] = np.nan

    roughness = compute_roughness(dem_filled)
    roughness[edge_mask] = np.nan

    vrm = compute_vrm(slope_deg, aspect_deg)
    vrm[edge_mask] = np.nan

    layers = {
        'slope': slope_deg,
        'aspect': aspect_deg,
        'tri': tri,
        'curvature': curv,
        'roughness': roughness,
        'vrm': vrm,
    }

    for name, data in layers.items():
        out = terrain_dir / f"{name}.tif"
        save_raster(data, profile, out)
        print(f"  [saved] terrain/{area_name}/{name}.tif")

    stamp_path.write_text(json.dumps(
        {'bbox': list(bbox), 'epsg': target_epsg,
         'edge_guard_px': EDGE_GUARD_PX,
         'terrain_version': TERRAIN_VERSION}, indent=2))

    print(f"[done] {area_name} — DEM + {len(layers)} derivatives")
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Download NASADEM and compute terrain derivatives\n'
                    'Usage: python scripts/03_process_dem.py cordillera_blanca\n'
                    '       python scripts/03_process_dem.py --all'
    )
    parser.add_argument('area_name', nargs='?', help='Single area name')
    parser.add_argument('--all', action='store_true', help='All areas')
    args = parser.parse_args()

    if not args.all and not args.area_name:
        parser.print_help()
        sys.exit(1)

    if args.all:
        areas = sorted(active_areas())
        for name, reason in EXCLUDED_AREAS.items():
            print(f"[excluded] {name} — {reason}")
        print(f"Processing DEM for {len(areas)} active areas")
        failed = []
        for name in areas:
            try:
                ok = process_one(name)
                if not ok:
                    failed.append(name)
            except Exception as e:
                print(f"[ERROR] {name}: {e}")
                failed.append(name)
        print(f"\n=== SUMMARY ===")
        print(f"Completed: {len(areas) - len(failed)}/{len(areas)}")
        if failed:
            print(f"Failed   : {failed}")
    else:
        ok = process_one(args.area_name)
        sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
