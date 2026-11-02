import sys
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask
from rasterio.windows import Window, transform as window_transform
from shapely.geometry import Point

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import EXPANDED_STUDY_AREAS, active_areas
from src.detection import RGI_PATH

LAKE_DIRS = {'sentinel2': 'lakes_s2', 'landsat': 'lakes_landsat'}
DEM_DIR = PROJECT_ROOT / 'data' / 'interim' / 'dem'
TERRAIN_DIR = PROJECT_ROOT / 'data' / 'interim' / 'terrain'
OUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'features'
OUT_GPKG = OUT_DIR / 'lake_features.gpkg'
OUT_CSV = OUT_DIR / 'lake_features.csv'

RING_INNER_M = 100.0
RING_OUTER_M = 500.0
DAM_PROBE_M = 100.0
LAYERS = ['slope', 'aspect', 'tri', 'curvature', 'roughness', 'vrm']
RING_COLUMNS = ['ring_slope_mean', 'ring_slope_p90', 'ring_relief_m', 'ring_aspect_north',
                'ring_aspect_east', 'ring_tri_mean', 'ring_curvature_mean',
                'ring_roughness_mean', 'ring_vrm_mean']


def load_lakes(source):
    frames = []
    areas = active_areas()
    for sensor, sub in LAKE_DIRS.items():
        if source not in (sensor, 'all'):
            continue
        d = PROJECT_ROOT / 'data' / 'processed' / sub
        files = sorted(f for a in areas for f in d.glob(f'{a}_*.gpkg')
                       if f.stem[len(a) + 1:].isdigit())
        print(f'  {sensor}: {len(files)} files')
        for f in files:
            g = gpd.read_file(f)
            g = g[g['area_name'].isin(areas)] if 'area_name' in g.columns else g
            if not g.empty:
                frames.append(g.to_crs(4326))
    if not frames:
        return gpd.GeoDataFrame()
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=4326)


def morphometrics(geom):
    area, perim = geom.area, geom.length
    hull = geom.convex_hull.area
    try:
        c = list(geom.minimum_rotated_rectangle.exterior.coords)
        sides = [np.hypot(c[i][0] - c[i - 1][0], c[i][1] - c[i - 1][1]) for i in range(1, 5)]
        elongation = max(sides) / min(sides) if min(sides) > 0 else np.nan
    except Exception:
        elongation = np.nan
    return {
        'area_m2': area,
        'perimeter_m': perim,
        'compactness': 4 * np.pi * area / perim ** 2 if perim > 0 else np.nan,
        'elongation': elongation,
        'fractal_dim': 2 * np.log(perim) / np.log(area) if area > 1 and perim > 1 else np.nan,
        'convexity': area / hull if hull > 0 else np.nan,
        'shore_dev_index': perim / (2 * np.sqrt(np.pi * area)) if area > 0 else np.nan,
    }


def read_layer(path):
    if not path.exists():
        return None
    with rasterio.open(path) as src:
        a = src.read(1).astype(np.float32)
        if src.nodata is not None:
            a[a == src.nodata] = np.nan
    return a


def window_mask(geom, transform, shape_hw, all_touched=True):
    h, w = shape_hw
    inv = ~transform
    minx, miny, maxx, maxy = geom.bounds
    c0, r0 = inv * (minx, maxy)
    c1, r1 = inv * (maxx, miny)
    cmin, cmax = max(0, int(np.floor(min(c0, c1)))), min(w, int(np.ceil(max(c0, c1))))
    rmin, rmax = max(0, int(np.floor(min(r0, r1)))), min(h, int(np.ceil(max(r0, r1))))
    if cmax <= cmin or rmax <= rmin:
        return None, None
    wt = window_transform(Window(cmin, rmin, cmax - cmin, rmax - rmin), transform)
    m = geometry_mask([geom], out_shape=(rmax - rmin, cmax - cmin), transform=wt,
                      invert=True, all_touched=all_touched)
    return (slice(rmin, rmax), slice(cmin, cmax)), m


def sample(arr, transform, xs, ys):
    cols, rows = (~transform) * (np.asarray(xs), np.asarray(ys))
    rows, cols = np.floor(rows).astype(int), np.floor(cols).astype(int)
    h, w = arr.shape
    ok = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    out = np.full(rows.shape, np.nan)
    out[ok] = arr[rows[ok], cols[ok]]
    return out


def dam_metrics(geom, dem, transform, lake_elev):
    ext = geom.exterior
    n = max(60, int(ext.length / 30))
    pts = [ext.interpolate(i / n, normalized=True) for i in range(n)]
    elev = sample(dem, transform, [p.x for p in pts], [p.y for p in pts])
    if not np.isfinite(elev).any():
        return np.nan, np.nan
    i = int(np.nanargmin(elev))
    outlet, outlet_elev = pts[i], float(elev[i])
    ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    tx = outlet.x + DAM_PROBE_M * np.cos(ang)
    ty = outlet.y + DAM_PROBE_M * np.sin(ang)
    outside = np.array([not geom.contains(Point(x, y)) for x, y in zip(tx, ty)])
    dv = sample(dem, transform, tx, ty)
    down = outside & np.isfinite(dv) & (dv < outlet_elev)
    freeboard = lake_elev - outlet_elev if np.isfinite(lake_elev) else np.nan
    drop = outlet_elev - float(np.min(dv[down])) if down.any() else np.nan
    return freeboard, drop


def terrain_features(gdf):
    out = pd.DataFrame(np.nan, index=gdf.index,
                       columns=['lake_elev_m', 'freeboard_m', 'outlet_drop_m'] + RING_COLUMNS)
    for area in gdf['area_name'].unique():
        dem_path = DEM_DIR / f'{area}_dem_utm.tif'
        if not dem_path.exists():
            print(f'    [WARN] no DEM for {area}')
            continue
        with rasterio.open(dem_path) as src:
            crs, transform, shape_hw = src.crs, src.transform, (src.height, src.width)
        dem = read_layer(dem_path)
        layers = {k: read_layer(TERRAIN_DIR / area / f'{k}.tif') for k in LAYERS}
        idx = gdf.index[gdf['area_name'] == area]
        sub = gdf.loc[idx].to_crs(crs)
        print(f'  terrain {area}: {len(idx)} polygons')
        for i, geom in zip(idx, sub.geometry):
            if geom is None or geom.is_empty:
                continue
            win, m = window_mask(geom, transform, shape_hw, all_touched=False)
            if win is not None and not m.any():
                win, m = window_mask(geom, transform, shape_hw, all_touched=True)
            if win is None:
                continue
            v = dem[win][m]
            lake_elev = float(np.nanmedian(v)) if np.isfinite(v).any() else np.nan
            out.at[i, 'lake_elev_m'] = lake_elev

            ring = geom.buffer(RING_OUTER_M).difference(geom.buffer(RING_INNER_M))
            win, m = window_mask(ring, transform, shape_hw)
            if win is not None:
                e = dem[win][m]
                e = e[np.isfinite(e)]
                if e.size:
                    out.at[i, 'ring_relief_m'] = float(e.max() - e.min())
                s = layers['slope']
                if s is not None:
                    s = s[win][m]
                    s = s[np.isfinite(s)]
                    if s.size:
                        out.at[i, 'ring_slope_mean'] = float(s.mean())
                        out.at[i, 'ring_slope_p90'] = float(np.percentile(s, 90))
                a = layers['aspect']
                if a is not None:
                    a = a[win][m]
                    a = np.radians(a[np.isfinite(a)])
                    if a.size:
                        out.at[i, 'ring_aspect_north'] = float(np.cos(a).mean())
                        out.at[i, 'ring_aspect_east'] = float(np.sin(a).mean())
                for k in ('tri', 'curvature', 'roughness', 'vrm'):
                    arr = layers[k]
                    if arr is None:
                        continue
                    x = arr[win][m]
                    x = x[np.isfinite(x)]
                    if x.size:
                        out.at[i, f'ring_{k}_mean'] = float(x.mean())

            fb, drop = dam_metrics(geom, dem, transform, lake_elev)
            out.at[i, 'freeboard_m'] = fb
            out.at[i, 'outlet_drop_m'] = drop
    return out


def glacier_distance(gdf):
    out = pd.Series(np.nan, index=gdf.index)
    if not RGI_PATH.exists():
        warnings.warn(f'RGI not found: {RGI_PATH}')
        return out
    for area in gdf['area_name'].unique():
        cfg = EXPANDED_STUDY_AREAS[area]
        pad = 0.2
        x0, y0, x1, y1 = cfg['bbox']
        gl = gpd.read_file(RGI_PATH, bbox=(x0 - pad, y0 - pad, x1 + pad, y1 + pad))
        if gl.empty:
            continue
        crs = f'EPSG:{cfg["epsg"]}'
        gl = gl.to_crs(crs)[['geometry']]
        gl = gl[gl.geometry.notna() & ~gl.geometry.is_empty]
        gl['geometry'] = gl.geometry.make_valid()
        idx = gdf.index[gdf['area_name'] == area]
        pts = gpd.GeoDataFrame(geometry=gdf.loc[idx].to_crs(crs).geometry.centroid, index=idx)
        j = gpd.sjoin_nearest(pts, gl, distance_col='d')
        out.loc[idx] = j[~j.index.duplicated()]['d'].reindex(idx).values
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', choices=['sentinel2', 'landsat', 'all'], default='all')
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    gdf = load_lakes(args.source)
    if gdf.empty:
        sys.exit('no lakes found: run 04_detect_lakes.py first')
    print(f'  lake-year observations: {len(gdf)}')

    morph = []
    for area in gdf['area_name'].unique():
        idx = gdf.index[gdf['area_name'] == area]
        crs = f'EPSG:{EXPANDED_STUDY_AREAS[area]["epsg"]}'
        geoms = gdf.loc[idx].to_crs(crs).geometry
        morph.append(pd.DataFrame([morphometrics(g) for g in geoms], index=idx))
    gdf = gdf.drop(columns=[c for c in ('area_m2',) if c in gdf.columns])
    gdf = gdf.join(pd.concat(morph))

    gdf = gdf.join(terrain_features(gdf))
    gdf['dist_glacier_m'] = glacier_distance(gdf)

    too_far = (gdf['dist_glacier_m'] > 5000 + 1).sum()
    if too_far:
        print(f'  [WARN] {too_far} lakes farther than 5 km from ice: check RGI version')

    gdf.to_file(OUT_GPKG, driver='GPKG')
    gdf.drop(columns='geometry').to_csv(OUT_CSV, index=False)
    num = gdf.select_dtypes('number')
    const = [c for c in num.columns if num[c].nunique(dropna=True) <= 1]
    print(f'  saved {len(gdf)} rows -> {OUT_GPKG.relative_to(PROJECT_ROOT)}')
    print(f'  missing (%):\n{(num.isna().mean() * 100).round(1).to_string()}')
    if const:
        print(f'  [WARN] constant columns: {const}')


if __name__ == '__main__':
    main()
