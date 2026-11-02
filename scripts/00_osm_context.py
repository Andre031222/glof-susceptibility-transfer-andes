import sys
import json
import time
import argparse
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import polygonize, unary_union

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import EXPANDED_STUDY_AREAS, active_areas

OUT_DIR = PROJECT_ROOT / 'data' / 'raw' / 'osm'
ENDPOINTS = ['https://overpass-api.de/api/interpreter',
             'https://overpass.kumi.systems/api/interpreter']
USER_AGENT = 'PanAndeanGLOF-research/1.0 (python urllib; glacial lake susceptibility)'
PAD_DEG = 0.1
TIMEOUT_S = 600
RETRIES = 4
PAUSE_S = 5
DAM_BUFFER_M = 50.0

TAG_KEYS = ['name', 'landuse', 'man_made', 'water', 'waterway', 'natural', 'industrial',
            'resource', 'highway', 'place', 'population']

ANTHRO_RULES = [
    ('man_made', {'tailings', 'tailings_pond', 'reservoir_covered', 'dam'}),
    ('water', {'tailings', 'tailings_pond', 'reservoir', 'basin'}),
    ('waterway', {'dam'}),
    ('landuse', {'reservoir', 'quarry', 'landfill', 'industrial'}),
]
DAM_TAGS = {'man_made=dam', 'waterway=dam'}

ROAD_CLASSES = ('motorway|motorway_link|trunk|trunk_link|primary|primary_link|secondary|'
                'secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|'
                'service|road|track')
PLACE_CLASSES = 'city|town|village|hamlet|isolated_dwelling'

QUERIES = {
    'anthropogenic': """
(
  nwr["landuse"~"^(quarry|industrial|reservoir|landfill)$"]({b});
  nwr["man_made"~"^(tailings|tailings_pond|reservoir_covered|dam)$"]({b});
  nwr["water"~"^(reservoir|basin|tailings|tailings_pond)$"]({b});
  nwr["waterway"="dam"]({b});
);
out tags geom;
""",
    'roads': f"""
(
  way["highway"~"^({ROAD_CLASSES})$"]({{b}});
);
out tags geom;
""",
    'settlements': f"""
(
  nwr["place"~"^({PLACE_CLASSES})$"]({{b}});
);
out tags center;
""",
}


def overpass(query):
    data = urllib.parse.urlencode({'data': query}).encode()
    last = None
    for attempt in range(RETRIES):
        for url in ENDPOINTS:
            req = urllib.request.Request(url, data=data, headers={'User-Agent': USER_AGENT})
            try:
                with urllib.request.urlopen(req, timeout=TIMEOUT_S + 60) as r:
                    out = json.loads(r.read())
                if 'remark' in out and 'error' in out['remark'].lower():
                    raise RuntimeError(out['remark'])
                return out['elements']
            except Exception as e:
                last = e
                print(f'    {url.split("/")[2]} failed ({type(e).__name__}: {str(e)[:120]})')
                time.sleep(PAUSE_S)
        wait = 30 * 2 ** attempt
        print(f'    retry {attempt + 1}/{RETRIES} in {wait} s')
        time.sleep(wait)
    raise RuntimeError(f'overpass failed: {last}')


def coords(geom):
    return [(p['lon'], p['lat']) for p in geom if p is not None]


def relation_geometry(el):
    rings = {'outer': [], 'inner': []}
    for m in el.get('members', []):
        if m.get('type') != 'way' or 'geometry' not in m:
            continue
        c = coords(m['geometry'])
        if len(c) >= 2:
            rings['inner' if m.get('role') == 'inner' else 'outer'].append(LineString(c))
    outer = list(polygonize(rings['outer']))
    if not outer:
        lines = rings['outer'] + rings['inner']
        return unary_union(lines) if lines else None
    shape = unary_union(outer)
    inner = list(polygonize(rings['inner']))
    if inner:
        shape = shape.difference(unary_union(inner))
    return shape if not shape.is_empty else None


def element_geometry(el):
    if el['type'] == 'node':
        return Point(el['lon'], el['lat'])
    if 'center' in el:
        return Point(el['center']['lon'], el['center']['lat'])
    if el['type'] == 'way' and 'geometry' in el:
        c = coords(el['geometry'])
        if len(c) >= 4 and c[0] == c[-1]:
            return Polygon(c)
        return LineString(c) if len(c) >= 2 else None
    if el['type'] == 'relation':
        return relation_geometry(el)
    return None


def to_frame(elements):
    rows = []
    for el in elements:
        geom = element_geometry(el)
        if geom is None or geom.is_empty:
            continue
        if not geom.is_valid:
            geom = geom.buffer(0)
        tags = el.get('tags', {})
        row = {'osm_type': el['type'], 'osm_id': int(el['id'])}
        row.update({k: tags.get(k) for k in TAG_KEYS})
        row['geometry'] = geom
        rows.append(row)
    cols = ['osm_type', 'osm_id'] + TAG_KEYS + ['geometry']
    if not rows:
        return gpd.GeoDataFrame({c: pd.Series(dtype=object) for c in cols[:-1]},
                                geometry=gpd.GeoSeries([], crs=4326), crs=4326)
    return gpd.GeoDataFrame(rows, columns=cols, geometry='geometry', crs=4326)


def anthro_tag(row):
    for key, values in ANTHRO_RULES:
        v = row.get(key)
        if isinstance(v, str) and v in values:
            return f'{key}={v}'
    return 'other'


def anthropogenic(df, epsg):
    if df.empty:
        return df.assign(tag=pd.Series(dtype=object))
    df = df.copy()
    df['tag'] = df.apply(anthro_tag, axis=1)
    utm = df.to_crs(epsg)
    dam = df['tag'].isin(DAM_TAGS)
    line = ~df.geom_type.isin(['Polygon', 'MultiPolygon'])
    grow = dam | line
    utm.loc[grow, 'geometry'] = utm.loc[grow].geometry.buffer(DAM_BUFFER_M)
    return utm.to_crs(4326)


def bbox_string(bbox):
    w, s, e, n = bbox
    return f'{s - PAD_DEG:.4f},{w - PAD_DEG:.4f},{n + PAD_DEG:.4f},{e + PAD_DEG:.4f}'


def download_area(area, force):
    out = OUT_DIR / f'{area}.gpkg'
    if out.exists() and not force:
        print(f'  {area}: exists, skipped')
        return
    cfg = EXPANDED_STUDY_AREAS[area]
    b = bbox_string(cfg['bbox'])
    layers = {}
    for layer, q in QUERIES.items():
        print(f'  {area}: querying {layer}')
        query = f'[out:json][timeout:{TIMEOUT_S}];' + q.replace('{b}', b)
        df = to_frame(overpass(query))
        if layer == 'anthropogenic':
            df = anthropogenic(df, cfg['epsg'])
        elif layer == 'roads':
            df = df[df.geom_type.isin(['LineString', 'MultiLineString'])]
        layers[layer] = df
        print(f'    {layer}: {len(df)} features')
        time.sleep(PAUSE_S)
    tmp = out.with_suffix('.part.gpkg')
    tmp.unlink(missing_ok=True)
    for layer, df in layers.items():
        df.to_file(tmp, layer=layer, driver='GPKG')
    tmp.replace(out)
    a = layers['anthropogenic']
    if len(a):
        print(a['tag'].value_counts().to_string())
    print(f'  {area}: saved {out.name}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('area', nargs='?')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    areas = active_areas()
    if args.area:
        if args.area not in EXPANDED_STUDY_AREAS:
            sys.exit(f'unknown area: {args.area}')
        areas = [args.area]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for area in areas:
        download_area(area, args.force)


if __name__ == '__main__':
    main()
