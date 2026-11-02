import sys
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import from_bounds, Window
from rasterio.warp import reproject, transform_bounds, Resampling
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / 'data' / 'interim' / 'basemap'

ZONES = {
    'peru_north': {'bbox': [-77.95, -10.70, -76.40, -8.70], 'epsg': 32718, 'months': (6, 7, 8)},
    'peru_south': {'bbox': [-72.75, -14.30, -70.30, -12.90], 'epsg': 32719, 'months': (6, 7, 8)},
    'bolivia': {'bbox': [-69.75, -16.75, -67.55, -13.70], 'epsg': 32719, 'months': (6, 7, 8)},
    'patagonia': {'bbox': [-74.25, -48.00, -72.35, -46.25], 'epsg': 32718, 'months': (1, 2, 3),
                  'cloud': 60, 'years': (2019, 2020, 2021, 2022, 2023, 2024)},
}
YEARS = (2022, 2023, 2024)
RES_M = 60.0
CLOUD_MAX = 20
MAX_SCENES = 60
SCENES_PER_TILE = 3
SCL_INVALID = (0, 1, 3, 8, 9, 10)
BANDS = ('B04', 'B03', 'B02')
BOA_OFFSET = 1000.0
GAMMA = 1.25
STRETCH = (1, 98.5)
FILL_PX = 12


def get_pc_client():
    import planetary_computer
    import pystac_client
    return pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                                     modifier=planetary_computer.sign_inplace)


def grid(bbox, epsg):
    l, b, r, t = transform_bounds('EPSG:4326', f'EPSG:{epsg}', *bbox)
    w, h = int(np.ceil((r - l) / RES_M)), int(np.ceil((t - b) / RES_M))
    return from_origin(l, t, RES_M, RES_M), w, h


def read_to_grid(href, bbox, dst_tr, w, h, epsg, resampling):
    import planetary_computer
    for attempt in range(3):
        try:
            with rasterio.open(planetary_computer.sign(href.split('?')[0])) as src:
                l, b, r, t = transform_bounds('EPSG:4326', src.crs, *bbox)
                win = from_bounds(l, b, r, t, transform=src.transform)
                win = win.intersection(Window(0, 0, src.width, src.height))
                if win.width < 1 or win.height < 1:
                    return None
                f = abs(src.transform.a) / RES_M
                oh, ow = max(1, int(round(win.height * f))), max(1, int(round(win.width * f)))
                data = src.read(1, window=win, out_shape=(oh, ow), resampling=resampling,
                                masked=False).astype(np.float32)
                tr = src.window_transform(win) * rasterio.Affine.scale(win.width / ow,
                                                                       win.height / oh)
                out = np.full((h, w), np.nan, dtype=np.float32)
                reproject(data, out, src_transform=tr, src_crs=src.crs, src_nodata=0,
                          dst_transform=dst_tr, dst_crs=f'EPSG:{epsg}', dst_nodata=np.nan,
                          resampling=resampling)
                return out
        except Exception:
            if attempt == 2:
                return None
    return None


def offset_for(item):
    base = str(item.properties.get('s2:processing_baseline', '0'))
    try:
        if float(base) >= 4.0:
            return BOA_OFFSET
    except ValueError:
        pass
    generated = item.id.split('_')[-1][:8]
    return BOA_OFFSET if generated.isdigit() and generated >= '20220125' else 0.0


def build(catalog, name, cfg):
    bbox, epsg = cfg['bbox'], cfg['epsg']
    tr, w, h = grid(bbox, epsg)
    items = []
    for y in cfg.get('years', YEARS):
        start = f'{y}-{cfg["months"][0]:02d}-01'
        end_m = cfg['months'][-1]
        end = f'{y}-{end_m:02d}-{30 if end_m in (6, 9, 11) else (28 if end_m == 2 else 31)}'
        items += list(catalog.search(collections=['sentinel-2-l2a'], bbox=bbox,
                                     datetime=f'{start}/{end}',
                                     query={'eo:cloud_cover': {'lt': cfg.get('cloud', CLOUD_MAX)}}).items())
    from shapely.geometry import shape, box
    zone = box(*bbox)

    def rank(it):
        cover = shape(it.geometry).intersection(zone).area / zone.area
        return cover * (1 - it.properties.get('eo:cloud_cover', 100) / 100)

    items.sort(key=rank, reverse=True)
    per_tile = {}
    for it in items:
        per_tile.setdefault(it.properties.get('s2:mgrs_tile', it.id), []).append(it)
    items = sorted([it for v in per_tile.values() for it in v[:SCENES_PER_TILE]], key=rank,
                   reverse=True)
    rgb = np.full((3, h, w), np.nan, dtype=np.float32)
    used = 0
    for it in items[:MAX_SCENES]:
        if np.isfinite(rgb[0]).mean() > 0.995:
            break
        scl = read_to_grid(it.assets['SCL'].href, bbox, tr, w, h, epsg, Resampling.nearest)
        if scl is None:
            continue
        ok = np.isfinite(scl) & ~np.isin(np.rint(np.nan_to_num(scl)), SCL_INVALID)
        need = ok & np.isnan(rgb[0])
        if need.sum() < 0.002 * w * h:
            continue
        with ThreadPoolExecutor(3) as pool:
            bands = list(pool.map(lambda b: read_to_grid(it.assets[b].href, bbox, tr, w, h, epsg,
                                                         Resampling.average), BANDS))
        if any(b is None for b in bands):
            continue
        off = offset_for(it)
        for k, b in enumerate(bands):
            v = (b - off) / 10000.0
            sel = need & np.isfinite(v)
            rgb[k][sel] = v[sel]
        used += 1
        print(f'  {name}: {it.id} cloud {it.properties.get("eo:cloud_cover", 0):.1f}%  '
              f'filled {100 * np.isfinite(rgb[0]).mean():.1f}%', flush=True)
    from rasterio.fill import fillnodata
    valid = np.isfinite(rgb[0])
    for k in range(3):
        band = np.nan_to_num(rgb[k], nan=0.0).astype(np.float32)
        rgb[k] = fillnodata(band, mask=valid.astype(np.uint8), max_search_distance=FILL_PX)
    filled = rgb[0] > 0
    img = np.zeros(rgb.shape, dtype=np.uint8)
    for k in range(3):
        v = rgb[k][filled]
        lo, hi = np.percentile(v, STRETCH)
        x = np.clip((rgb[k] - lo) / (hi - lo), 0, 1) ** (1 / GAMMA)
        img[k] = (x * 254 + 1).astype(np.uint8)
    img[:, ~filled] = 0
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f'{name}_s2_rgb.tif'
    prof = {'driver': 'GTiff', 'height': h, 'width': w, 'count': 3, 'dtype': 'uint8',
            'crs': f'EPSG:{epsg}', 'transform': tr, 'nodata': 0, 'compress': 'deflate',
            'photometric': 'RGB'}
    with rasterio.open(out, 'w', **prof) as dst:
        dst.write(img)
    print(f'[done] {name}: {used} scenes, {w}x{h} px at {RES_M:.0f} m -> {out.name}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('zones', nargs='*', default=list(ZONES))
    args = ap.parse_args()
    catalog = get_pc_client()
    for z in args.zones:
        if z not in ZONES:
            sys.exit(f'unknown zone {z}')
        build(catalog, z, ZONES[z])


if __name__ == '__main__':
    main()
