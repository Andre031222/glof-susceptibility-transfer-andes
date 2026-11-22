import sys
import json
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.windows import from_bounds
from rasterio.warp import transform_bounds, reproject, Resampling
from rasterio.transform import from_origin

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
OUT_DIR = PROJECT_ROOT / 'data' / 'interim' / 'events'
FIG_DIR = PROJECT_ROOT / 'paper' / 'figures'
RES_M = 10.0
MAX_TRY = 25
BAD_MAX = 0.08
SCL_BAD = (0, 1, 3, 8, 9, 10)

EVENTS = [
    {'key': 'humantay', 'lake_phys': 'cordillera_urubamba_P00026', 'label': 'Humantay (moraine dam)',
     'note': 'moraine dam, GLOF 23 Feb 2020', 'event_year': 2020, 'pre': ('2018-05-01', '2020-02-20'),
     'post': ('2020-02-24', '2021-12-31'), 'epsg': 32718, 'half_km': 1.0, 'cloud': 60},
    {'key': 'cachet', 'lake_phys': 'patagonia_npi_P02405', 'label': 'Lago Cachet 2 (ice dam)',
     'note': 'ice dam, repeated GLOFs', 'event_year': None, 'pre': ('2017-01-01', '2024-12-31'),
     'post': ('2017-01-01', '2024-12-31'), 'epsg': 32718, 'half_km': 2.6, 'cloud': 40},
]

def load(name, file):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def chip_grid(lon, lat, epsg, half_km):
    from pyproj import Transformer
    x, y = Transformer.from_crs('EPSG:4326', f'EPSG:{epsg}', always_xy=True).transform(lon, lat)
    half = half_km * 1000
    n = int(2 * half / RES_M)
    return from_origin(x - half, y + half, RES_M, RES_M), n, (x - half, y - half, x + half, y + half)


def read_band(href, bounds, epsg, tr, n, resampling=Resampling.bilinear):
    import planetary_computer
    with rasterio.open(planetary_computer.sign(href.split('?')[0])) as src:
        l, b, r, t = transform_bounds(f'EPSG:{epsg}', src.crs, *bounds)
        win = from_bounds(l, b, r, t, transform=src.transform).round_offsets().round_lengths()
        data = src.read(1, window=win, boundless=True, fill_value=0).astype(np.float32)
        out = np.full((n, n), np.nan, dtype=np.float32)
        reproject(data, out, src_transform=src.window_transform(win), src_crs=src.crs, src_nodata=0,
                  dst_transform=tr, dst_crs=f'EPSG:{epsg}', dst_nodata=np.nan, resampling=resampling)
        return out


def mosaic(items, asset, bounds, epsg, tr, n, resampling):
    out = None
    for it in items:
        a = read_band(it.assets[asset].href, bounds, epsg, tr, n, resampling)
        if asset != 'SCL':
            a = a - load('b18', '18_basemap_rgb.py').offset_for(it)
        out = a if out is None else np.where(np.isnan(out), a, out)
    return out


def candidates(catalog, lon, lat, dates, cloud, epsg, tr, n, bounds, lake_mask):
    items = catalog.search(collections=['sentinel-2-l2a'], intersects={'type': 'Point', 'coordinates': [lon, lat]},
                           datetime=f'{dates[0]}/{dates[1]}',
                           query={'eo:cloud_cover': {'lt': cloud}}).item_collection()
    by_day = {}
    for it in items:
        by_day.setdefault(it.datetime.strftime('%Y-%m-%d'), []).append(it)
    days = sorted(by_day.items(), key=lambda kv: max(i.properties['eo:cloud_cover'] for i in kv[1]))
    out = []
    for day, its in days[:MAX_TRY]:
        scl = mosaic(its, 'SCL', bounds, epsg, tr, n, Resampling.nearest)
        bad = np.isnan(scl) | np.isin(scl, SCL_BAD)
        if bad.mean() > BAD_MAX or bad[lake_mask].mean() > 0.05:
            continue
        out.append((its, day, float((scl[lake_mask] == 6).mean())))
    return out


def rgb_of(items, bounds, epsg, tr, n):
    rgb = np.stack([mosaic(items, k, bounds, epsg, tr, n, Resampling.bilinear) for k in ('B04', 'B03', 'B02')])
    return np.clip(rgb / 10000.0, 0, 1)


def fetch():
    import pystac_client
    import planetary_computer
    from rasterio.features import rasterize
    catalog = pystac_client.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1',
                                        modifier=planetary_computer.sign_inplace)
    phys = gpd.read_file(PROJECT_ROOT / 'data' / 'processed' / 'physical' / 'lake_footprints.gpkg')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    meta = {}
    for ev in EVENTS:
        g = phys[phys['lake_phys'] == ev['lake_phys']].to_crs(4326)
        c = g.geometry.union_all().centroid
        tr, n, bounds = chip_grid(c.x, c.y, ev['epsg'], ev['half_km'])
        gl = g.to_crs(ev['epsg'])
        mask = rasterize([(gl.geometry.union_all(), 1)], out_shape=(n, n), transform=tr).astype(bool)
        meta[ev['key']] = {'lon': c.x, 'lat': c.y, 'bounds': bounds}
        pre = candidates(catalog, c.x, c.y, ev['pre'], ev['cloud'], ev['epsg'], tr, n, bounds, mask)
        post = pre if ev['post'] == ev['pre'] else candidates(catalog, c.x, c.y, ev['post'], ev['cloud'],
                                                              ev['epsg'], tr, n, bounds, mask)
        if not pre or not post:
            print(f"  {ev['key']}: no clear scenes ({len(pre)} pre, {len(post)} post)")
            continue
        full = max(pre, key=lambda x: (round(x[2], 1), x[1]))
        low = min(post, key=lambda x: (round(x[2], 1), x[1]))
        for when, (its, day, wf) in (('pre', full), ('post', low)):
            np.save(OUT_DIR / f"{ev['key']}_{when}.npy", rgb_of(its, bounds, ev['epsg'], tr, n))
            meta[ev['key']][when] = {'scenes': [i.id for i in its], 'date': day, 'water_frac': wf}
            print(f"  {ev['key']} {when}: {day} ({len(its)} tiles) water {wf:.2f}")
        gl.to_file(OUT_DIR / f"{ev['key']}_lake.gpkg")
    (OUT_DIR / 'scenes.json').write_text(json.dumps(meta, indent=1))


def stretch(rgb):
    lo, hi = np.nanpercentile(rgb, (1, 99), axis=(1, 2), keepdims=True)
    return np.clip((rgb - lo) / np.maximum(hi - lo, 1e-6), 0, 1) ** (1 / 1.2)


def figure():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.titlesize': 8,
                         'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5})
    meta = json.loads((OUT_DIR / 'scenes.json').read_text())
    fig, axes = plt.subplots(len(EVENTS), 2, figsize=(5.51, 5.6), gridspec_kw={'wspace': 0.05, 'hspace': 0.16})
    letters = iter('abcd')
    for row, ev in zip(axes, EVENTS):
        m = meta[ev['key']]
        lake = gpd.read_file(OUT_DIR / f"{ev['key']}_lake.gpkg")
        l, b, r, t = m['bounds']
        names = ('Before', 'After') if ev['event_year'] else ('Full', 'Drained')
        bar = 500 if ev['half_km'] < 1.5 else 1000
        for ax, when, name in zip(row[:2], ('pre', 'post'), names):
            ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(np.moveaxis(stretch(np.load(OUT_DIR / f"{ev['key']}_{when}.npy")), 0, -1),
                      extent=(l, r, b, t))
            lake.boundary.plot(ax=ax, color='#ffd400', linewidth=0.6, linestyle=(0, (3, 2)))
            ax.set_xlim(l, r); ax.set_ylim(b, t)
            pad = (r - l) * 0.05
            ax.plot([r - pad - bar, r - pad], [b + pad, b + pad], color='white', linewidth=2)
            ax.text(r - pad - bar / 2, b + pad * 1.6, f'{bar / 1000:g} km', color='white', ha='center',
                    va='bottom', fontsize=7.5)
            ax.set_title(f"({next(letters)}) {name}, {m[when]['date']}", loc='left')
        row[0].set_ylabel(ev["label"], fontsize=8)
    out = FIG_DIR / 'Fig09_event_evidence.png'
    fig.savefig(out, dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    print(f'saved {out.name}')


if __name__ == '__main__':
    if '--figure-only' not in sys.argv:
        fetch()
    figure()
