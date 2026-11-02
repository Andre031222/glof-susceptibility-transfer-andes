import argparse
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import box

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import EXPANDED_STUDY_AREAS, active_areas
from src.detection import MIN_AREA_M2, GLACIER_BUFFER_M, RGI_PATH

DATA_ROOT = Path(os.environ.get('GLOF_ROOT') or PROJECT_ROOT)
REF_DIR = PROJECT_ROOT / 'data' / 'raw' / 'reference_inventories'
COUNTRIES = PROJECT_ROOT / 'data' / 'raw' / 'boundaries' / 'ne_10m_admin_0_countries.zip'
SENSOR_DIRS = {'s2': 'lakes_s2', 'landsat': 'lakes_landsat'}
SIZE_BINS = [0.0, 2e4, 1e5, 1e6, np.inf]
SIZE_LABELS = ['<2 ha', '2-10 ha', '10-100 ha', '>100 ha']


def ref_frame(geoms, years, lo, hi, ids, crs):
    g = gpd.GeoDataFrame({'ref_id': ids, 'ref_year': years, 'year_lo': lo, 'year_hi': hi},
                         geometry=gpd.GeoSeries(geoms, crs=crs).force_2d().values, crs=crs)
    g = g[g.geometry.notna() & ~g.geometry.is_empty].copy()
    g['geometry'] = g.geometry.make_valid()
    return g.to_crs(4326).reset_index(drop=True)


def load_macmanaway(path=None, **_):
    root = Path(path) if path else REF_DIR / 'macmanaway2025_bolivia' / 'Lakes'
    parts = []
    for shp in sorted(root.glob('*/*-lakes.shp')):
        y = int(shp.parent.name)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            g = gpd.read_file(shp)
        ids = [f'{y}_{v}' for v in (g['id'] if 'id' in g.columns else g.index)]
        parts.append(ref_frame(g.geometry.values, y, y, y, ids, g.crs))
    if not parts:
        sys.exit(f'[ERROR] no MacManaway shapefiles under {root}')
    return pd.concat(parts, ignore_index=True)


def load_wood(path=None, **_):
    f = Path(path) if path else REF_DIR / 'wood2021_peru' / 'mmc2' / 'projectGLOP_lakes.kml'
    if not f.exists():
        sys.exit(f'[ERROR] {f} not found')
    g = gpd.read_file(f)
    ids = [f'glop_{i:05d}' for i in range(len(g))]
    return ref_frame(g.geometry.values, 2019, 2017, 2019, ids, g.crs or 'EPSG:4326')


def load_zhang(path=None, **_):
    f = Path(path) if path else REF_DIR / 'zhang2024_global_via_veh2025' / 'zhang2024_andes_1990_2020.gpkg'
    if not f.exists():
        sys.exit(f'[ERROR] {f} not found; build it from Lakes19902020_damtype.RDS first')
    g = gpd.read_file(f)
    y = g['Year_new'].round().astype(int).values
    ids = [f'z{int(u)}' for u in g['UniqueID']] if 'UniqueID' in g.columns else list(range(len(g)))
    return ref_frame(g.geometry.values, y, y, y, ids, g.crs)


def load_shugar(path=None, **_):
    root = Path(path) if path else REF_DIR / 'shugar2020_hma_gli'
    parts = []
    for shp in sorted(root.glob('*.shp')):
        m = re.search(r'_(\d{4})_(\d{2})\.shp$', shp.name)
        if not m:
            continue
        lo = int(m.group(1))
        hi = lo // 100 * 100 + int(m.group(2))
        hi = hi + 100 if hi < lo else hi
        g = gpd.read_file(shp)
        ids = [f'{lo}_{i}' for i in range(len(g))]
        parts.append(ref_frame(g.geometry.values, int(round((lo + hi) / 2)), lo, hi, ids, g.crs))
    if not parts:
        sys.exit(f'[ERROR] no HMA_GLI shapefiles under {root} (NSIDC requires an Earthdata login)')
    return pd.concat(parts, ignore_index=True)


def load_file(path=None, year=None, year_column=None, **_):
    if not path:
        sys.exit('[ERROR] --ref-path is required with --reference file')
    g = gpd.read_file(path)
    if year_column:
        y = g[year_column].round().astype(int).values
    elif year is not None:
        y = int(year)
    else:
        sys.exit('[ERROR] give --ref-year or --ref-year-column with --reference file')
    ids = [f'f{i}' for i in range(len(g))]
    return ref_frame(g.geometry.values, y, y, y, ids, g.crs or 'EPSG:4326')


REFERENCES = {
    'wood2021': {
        'loader': load_wood, 'doi': '10.1016/j.gloplacha.2021.103574',
        'buffer_m': 3000.0, 'country': 'Peru',
        'note': 'projectGLOP KML supplement; manual digitisation on 2019 Landsat (2017-2018 where cloudy); lakes within 3 km of RGI 6.0',
    },
    'macmanaway2025': {
        'loader': load_macmanaway, 'doi': '10.1017/jog.2025.10112',
        'buffer_m': 2000.0, 'country': 'Bolivia',
        'note': 'Zenodo 10.5281/zenodo.22043439 (CC-BY-4.0); annual 2016-2022; lakes within 2 km of glaciers',
    },
    'zhang2024': {
        'loader': load_zhang, 'doi': '10.1038/s43247-024-01544-y',
        'buffer_m': 10000.0, 'country': None,
        'note': 'global 1990/2020 outlines, redistributed in Zenodo 10.5281/zenodo.17896427 (CC-BY-4.0); lakes hydraulically connected to glaciers within 10 km, >= 0.002 km2',
    },
    'shugar2020': {
        'loader': load_shugar, 'doi': '10.1038/s41558-020-0855-4',
        'buffer_m': 1000.0, 'country': None,
        'note': 'NSIDC HMA_GLI v1 (10.5067/UO20NYM3YQB4); needs Earthdata login; ice-proximal lakes within 1 km, >= 0.05 km2',
    },
    'file': {
        'loader': load_file, 'doi': '', 'buffer_m': GLACIER_BUFFER_M, 'country': None,
        'note': 'user-supplied polygon file',
    },
}


def summarise(ref, name):
    utm = ref.to_crs(ref.estimate_utm_crs())
    a = utm.geometry.area
    print(f'=== reference {name} ===')
    print(f'  lakes: {len(ref)}')
    yr = ref.groupby(['ref_year', 'year_lo', 'year_hi']).size()
    print('  years (ref_year, lo, hi -> n):')
    for k, v in yr.items():
        print(f'    {k} -> {v}')
    x0, y0, x1, y1 = ref.total_bounds
    print(f'  bbox: lon [{x0:.3f}, {x1:.3f}] lat [{y0:.3f}, {y1:.3f}]')
    q = a.quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    print('  area m2 quantiles: ' + ', '.join(f'q{int(100 * k)}={v:,.0f}' for k, v in q.items()))
    print(f'  lakes >= {MIN_AREA_M2:.0f} m2: {int((a >= MIN_AREA_M2).sum())} '
          f'({100 * (a >= MIN_AREA_M2).mean():.1f}%)')
    cent = ref.geometry.representative_point()
    rows = []
    for area, cfg in EXPANDED_STUDY_AREAS.items():
        inside = cent.within(box(*cfg['bbox']))
        if inside.any():
            rows.append((area, int(inside.sum()),
                         int((inside & (a.values >= MIN_AREA_M2)).sum()),
                         sorted(ref.loc[inside, 'ref_year'].unique().tolist())))
    print('  study areas covered (all lakes, lakes >= min area, years):')
    for r in rows:
        print(f'    {r[0]:26s} {r[1]:6d} {r[2]:6d} {r[3]}')
    if not rows:
        print('    none')


def country_geom(name):
    if not name:
        return None
    if not COUNTRIES.exists():
        warnings.warn(f'{COUNTRIES} not found; country clip skipped')
        return None
    c = gpd.read_file(f'zip://{COUNTRIES}')
    c = c[c['ADMIN'] == name]
    return None if c.empty else c.to_crs(4326).geometry.union_all()


def domain_polygon(cfg, epsg, buffer_m, country):
    dom = box(*cfg['bbox'])
    if country is not None:
        dom = dom.intersection(country)
    dom = gpd.GeoSeries([dom], crs=4326).to_crs(epsg).iloc[0]
    if not RGI_PATH.exists():
        warnings.warn(f'{RGI_PATH} not found; glacier-distance rule skipped')
        return dom
    pad = buffer_m / 111000.0 * 1.5
    b = cfg['bbox']
    gl = gpd.read_file(RGI_PATH, bbox=(b[0] - pad, b[1] - pad, b[2] + pad, b[3] + pad))
    if gl.empty:
        return dom.intersection(box(0, 0, 0, 0))
    geom = gl.to_crs(epsg).geometry
    geom = geom[geom.notna() & ~geom.is_empty].make_valid()
    return dom.intersection(geom.buffer(buffer_m).union_all())


def elevation_ok(points, area_name, cfg):
    dem = PROJECT_ROOT / 'data' / 'interim' / 'dem' / f'{area_name}_dem_utm.tif'
    if not dem.exists():
        return np.ones(len(points), dtype=bool)
    import rasterio
    with rasterio.open(dem) as src:
        pts = points.to_crs(src.crs)
        vals = np.array([v[0] for v in src.sample(list(zip(pts.x, pts.y)))], dtype=float)
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    ok = (vals >= cfg['elev_min_m']) & (vals <= cfg['elev_max_m'])
    return ok | np.isnan(vals)


def size_class(a):
    return pd.cut(pd.Series(a, dtype=float), SIZE_BINS, labels=SIZE_LABELS, right=False).astype(str).values


def compare(ref, det, domain, min_area=MIN_AREA_M2):
    ref = ref.reset_index(drop=True).copy()
    det = det.reset_index(drop=True).copy()
    ref['area_m2'] = ref.geometry.area
    det['area_m2'] = det.geometry.area
    if domain is None:
        r_in = np.ones(len(ref), dtype=bool)
        d_in = np.ones(len(det), dtype=bool)
    else:
        r_in = ref.geometry.centroid.within(domain).values
        d_in = det.geometry.centroid.within(domain).values
    ref['eligible'] = r_in & (ref['area_m2'].values >= min_area)
    det['eligible'] = d_in & (det['area_m2'].values >= min_area)
    if len(ref) and len(det):
        ri, di = det.sindex.query(ref.geometry, predicate='intersects')
    else:
        ri = di = np.array([], dtype=int)
    ref['matched'] = np.isin(np.arange(len(ref)), ri)
    det['matched'] = np.isin(np.arange(len(det)), di)
    n = len(ref) + len(det)
    adj = coo_matrix((np.ones(len(ri)), (ri, di + len(ref))), shape=(n, n))
    _, comp = connected_components(adj, directed=False)
    ref['component'] = comp[:len(ref)]
    det['component'] = comp[len(ref):]
    ref['size_class'] = size_class(ref['area_m2'].values)
    det['size_class'] = size_class(det['area_m2'].values)
    keep = set(ref.loc[ref['eligible'] & ref['matched'], 'component'])
    ra = ref[ref['component'].isin(keep)].groupby('component')['area_m2'].agg(['sum', 'size'])
    da = det[det['component'].isin(keep)].groupby('component')['area_m2'].agg(['sum', 'size'])
    pairs = pd.DataFrame({'area_ref_m2': ra['sum'], 'n_ref': ra['size']}).join(
        pd.DataFrame({'area_det_m2': da['sum'], 'n_det': da['size']}), how='inner')
    pairs['size_class'] = size_class(pairs['area_ref_m2'].values)
    pairs = pairs.reset_index()
    return ref, det, pairs


def metrics(ref, det, pairs):
    er = ref[ref['eligible']]
    ed = det[det['eligible']]
    n_ref, k_ref = len(er), int(er['matched'].sum())
    n_det, k_det = len(ed), int(ed['matched'].sum())
    rec = k_ref / n_ref if n_ref else np.nan
    pre = k_det / n_det if n_det else np.nan
    f1 = 2 * pre * rec / (pre + rec) if n_ref and n_det and (pre + rec) > 0 else np.nan
    out = {'n_ref': n_ref, 'n_ref_matched': k_ref, 'recall': rec,
           'n_det': n_det, 'n_det_matched': k_det, 'precision': pre, 'f1': f1,
           'n_pairs': len(pairs)}
    if len(pairs):
        d = pairs['area_det_m2'] - pairs['area_ref_m2']
        out.update({'area_bias_m2': d.mean(),
                    'rel_area_bias_pct': 100 * d.sum() / pairs['area_ref_m2'].sum(),
                    'area_mae_m2': d.abs().mean(),
                    'median_ape_pct': 100 * (d.abs() / pairs['area_ref_m2']).median()})
    else:
        out.update({'area_bias_m2': np.nan, 'rel_area_bias_pct': np.nan,
                    'area_mae_m2': np.nan, 'median_ape_pct': np.nan})
    return out


def metric_rows(ref, det, pairs, keys):
    rows = [{**keys, 'size_class': 'all', **metrics(ref, det, pairs)}]
    for sc in SIZE_LABELS:
        r = ref[ref['size_class'] == sc]
        d = det[det['size_class'] == sc]
        p = pairs[pairs['size_class'] == sc]
        if r['eligible'].any() or d['eligible'].any():
            rows.append({**keys, 'size_class': sc, **metrics(r, d, p)})
    return rows


def detection_years(area, sensor):
    d = DATA_ROOT / 'data' / 'processed' / SENSOR_DIRS[sensor]
    out = {}
    for f in d.glob(f'{area}_*.gpkg'):
        m = re.fullmatch(rf'{re.escape(area)}_(\d{{4}})\.gpkg', f.name)
        if m:
            out[int(m.group(1))] = f
    return out


def pick_year(years, ref_year, lo, hi, tol):
    cand = [y for y in years if lo - tol <= y <= hi + tol]
    if not cand:
        return None
    return sorted(cand, key=lambda y: (abs(y - ref_year), -y))[0]


def run(ref, name, spec, areas, sensors, tol, buffer_m):
    country = country_geom(spec['country'])
    buf = min(GLACIER_BUFFER_M, buffer_m)
    rows, lakes = [], []
    pooled = {}
    for area in areas:
        cfg = EXPANDED_STUDY_AREAS[area]
        sub = ref[ref.geometry.intersects(box(*cfg['bbox']))]
        if sub.empty:
            continue
        epsg = cfg['epsg']
        domain = None
        for sensor in sensors:
            years = detection_years(area, sensor)
            if not years:
                continue
            for (ry, lo, hi), grp in sub.groupby(['ref_year', 'year_lo', 'year_hi']):
                dy = pick_year(years, ry, lo, hi, tol)
                if dy is None:
                    continue
                if domain is None:
                    domain = domain_polygon(cfg, epsg, buf, country)
                det = gpd.read_file(years[dy])
                det = det[det.geometry.notna() & ~det.geometry.is_empty].to_crs(epsg)
                r = grp.to_crs(epsg)
                r = r[elevation_ok(r.geometry.centroid, area, cfg)]
                rr, dd, pp = compare(r, det, domain)
                keys = {'reference': name, 'sensor': sensor, 'area_name': area,
                        'ref_year': int(ry), 'det_year': int(dy)}
                rows.extend(metric_rows(rr, dd, pp, keys))
                tag = f'{area}|{ry}|{dy}|'
                for kind, t in (('reference', rr), ('detection', dd)):
                    idc = 'ref_id' if kind == 'reference' else 'lake_id'
                    c = gpd.GeoSeries(t.geometry.centroid, crs=epsg).to_crs(4326)
                    lakes.append(pd.DataFrame({
                        **keys, 'kind': kind,
                        'lake_id': t[idc].values if idc in t.columns else t.index.values,
                        'area_m2': t['area_m2'].round(1).values, 'size_class': t['size_class'].values,
                        'eligible': t['eligible'].values, 'matched': t['matched'].values,
                        'component': [tag + str(v) for v in t['component'].values],
                        'lon': c.x.round(6).values, 'lat': c.y.round(6).values}))
                pp = pp.assign(component=tag + pp['component'].astype(str))
                acc = pooled.setdefault(sensor, ([], [], []))
                acc[0].append(rr.drop(columns='geometry'))
                acc[1].append(dd.drop(columns='geometry'))
                acc[2].append(pp)
                mm = metrics(rr, dd, pp)
                print(f'  {sensor:7s} {area:24s} ref {ry} <-> det {dy}: n_ref={mm["n_ref"]} '
                      f'n_det={mm["n_det"]} recall={mm["recall"]:.3f} precision={mm["precision"]:.3f}')
    for sensor, (rs, ds, ps) in pooled.items():
        keys = {'reference': name, 'sensor': sensor, 'area_name': 'ALL', 'ref_year': -1, 'det_year': -1}
        rows.extend(metric_rows(pd.concat(rs), pd.concat(ds), pd.concat(ps), keys))
    return pd.DataFrame(rows), (pd.concat(lakes, ignore_index=True) if lakes else pd.DataFrame())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--reference', required=True, choices=sorted(REFERENCES))
    ap.add_argument('--ref-path')
    ap.add_argument('--ref-year', type=int)
    ap.add_argument('--ref-year-column')
    ap.add_argument('--ref-buffer-m', type=float)
    ap.add_argument('--areas', nargs='*')
    ap.add_argument('--sensor', choices=['s2', 'landsat', 'all'], default='all')
    ap.add_argument('--year-tol', type=int, default=1)
    ap.add_argument('--summary', action='store_true')
    ap.add_argument('--out-dir', default=str(DATA_ROOT / 'models'))
    a = ap.parse_args(argv)

    spec = REFERENCES[a.reference]
    ref = spec['loader'](path=a.ref_path, year=a.ref_year, year_column=a.ref_year_column)
    print(f'{a.reference}: doi {spec["doi"] or "-"}; {spec["note"]}')
    summarise(ref, a.reference)
    if a.summary:
        return
    areas = a.areas or active_areas()
    sensors = ['s2', 'landsat'] if a.sensor == 'all' else [a.sensor]
    buffer_m = a.ref_buffer_m if a.ref_buffer_m is not None else spec['buffer_m']
    print(f'domain: study bbox, country={spec["country"]}, centroid within '
          f'{min(GLACIER_BUFFER_M, buffer_m):.0f} m of RGI 7.0, reference lakes >= {MIN_AREA_M2:.0f} m2, '
          f'year tolerance +-{a.year_tol}')
    table, lakes = run(ref, a.reference, spec, areas, sensors, a.year_tol, buffer_m)
    if table.empty:
        print('No detection files overlap this reference in space and time '
              f'(looked in {DATA_ROOT / "data" / "processed"}); nothing written.')
        return
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / 'inventory_validation.csv'
    if f.exists():
        old = pd.read_csv(f)
        table = pd.concat([old[old['reference'] != a.reference], table], ignore_index=True)
    table.round(4).to_csv(f, index=False)
    m = out / f'inventory_validation_matches_{a.reference}.csv'
    lakes.to_csv(m, index=False)
    print(f'wrote {f}')
    print(f'wrote {m}')
    print(table[(table['reference'] == a.reference) & (table['area_name'] == 'ALL')]
          .drop(columns=['reference', 'area_name', 'ref_year', 'det_year']).round(3).to_string(index=False))


if __name__ == '__main__':
    main()
