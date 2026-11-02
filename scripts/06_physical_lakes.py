import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import rasterio
from rasterio.features import rasterize
from sklearn.cluster import DBSCAN

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config_expanded_study_areas import EXPANDED_STUDY_AREAS, analysis_region

FEATURES_GPKG = PROJECT_ROOT / 'data' / 'processed' / 'features' / 'lake_features.gpkg'
OUT_DIR = PROJECT_ROOT / 'data' / 'processed' / 'physical'
MAP_FILE = OUT_DIR / 'lake_physical_map.csv'
FOOTPRINTS = OUT_DIR / 'lake_footprints.gpkg'
VALIDITY_FILE = OUT_DIR / 'lake_year_validity.csv'
RAW = {'landsat': PROJECT_ROOT / 'data' / 'raw' / 'landsat',
       'sentinel2': PROJECT_ROOT / 'data' / 'raw' / 'sentinel2'}

COUNTRIES_FILE = PROJECT_ROOT / 'data' / 'raw' / 'boundaries' / 'ne_10m_admin_0_countries.zip'
COUNTRIES_URL = 'https://naciscdn.org/naturalearth/10m/cultural/ne_10m_admin_0_countries.zip'
BASINS_DIR = PROJECT_ROOT / 'data' / 'raw' / 'hydrobasins'
BASINS_URL = 'https://data.hydrosheds.org/file/HydroBASINS/standard/hybas_sa_lev{lev:02d}_v1c.zip'
BASIN_LEVELS = (8, 10, 12)
OSM_DIR = PROJECT_ROOT / 'data' / 'raw' / 'osm'
ANTHRO_DIST_M = 100.0
OSM_COLS = ['anthropogenic', 'anthropogenic_tag', 'dist_road_m', 'dist_settlement_m']

EPS_M = 200.0
VALID_FRAC = 0.8


def cluster_lakes(g):
    parts = []
    for area, sub in g.groupby('area_name'):
        epsg = EXPANDED_STUDY_AREAS[area]['epsg']
        parts.append(sub.assign(epsg=epsg))
    g = pd.concat(parts)
    out = []
    for epsg, sub in g.groupby('epsg'):
        utm = sub.to_crs(epsg)
        cent = utm.geometry.centroid
        labels = DBSCAN(eps=EPS_M, min_samples=1).fit_predict(np.c_[cent.x, cent.y])
        out.append(pd.DataFrame({
            'lake_id': sub['lake_id'].values,
            'acquisition': sub['area_name'].values,
            'year': sub['year'].values,
            'cluster': [f'{epsg}_{l}' for l in labels],
            'cx': cent.x.values, 'cy': cent.y.values, 'epsg': epsg,
        }))
    return pd.concat(out, ignore_index=True)


def name_lakes(phys):
    phys['analysis_region'] = phys['acquisition'].map(analysis_region)
    region = (phys.groupby(['cluster', 'analysis_region']).size().rename('n').reset_index()
              .sort_values(['cluster', 'n'], ascending=[True, False])
              .drop_duplicates('cluster').set_index('cluster')['analysis_region'])
    first = phys.drop_duplicates('cluster').sort_values('cluster')[['cluster']].copy()
    first['analysis_region'] = first['cluster'].map(region)
    first['seq'] = first.groupby('analysis_region').cumcount()
    names = dict(zip(first['cluster'],
                     first['analysis_region'] + '_P' + first['seq'].map('{:05d}'.format)))
    phys['analysis_region'] = phys['cluster'].map(region)
    phys['lake_phys'] = phys['cluster'].map(names)
    return phys.drop(columns='cluster')


def fetch(url, path):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as r, open(path, 'wb') as f:
        while chunk := r.read(1 << 20):
            f.write(chunk)


def countries(points):
    if not COUNTRIES_FILE.exists():
        COUNTRIES_FILE.parent.mkdir(parents=True, exist_ok=True)
        fetch(COUNTRIES_URL, COUNTRIES_FILE)
    c = gpd.read_file(COUNTRIES_FILE, columns=['NAME_EN']).rename(columns={'NAME_EN': 'country'})
    j = gpd.sjoin_nearest(points.to_crs(3857), c.to_crs(3857), how='left')
    return j[~j.index.duplicated()]['country']


def basins(points):
    out = pd.DataFrame(index=points.index)
    BASINS_DIR.mkdir(parents=True, exist_ok=True)
    bbox = tuple(points.total_bounds + np.array([-0.5, -0.5, 0.5, 0.5]))
    for lev in BASIN_LEVELS:
        path = BASINS_DIR / f'hybas_sa_lev{lev:02d}_v1c.zip'
        if not path.exists():
            fetch(BASINS_URL.format(lev=lev), path)
        b = gpd.read_file(path, bbox=bbox, columns=['HYBAS_ID']).to_crs(4326)
        j = gpd.sjoin(points, b, how='left', predicate='within')
        j = j[~j.index.duplicated()]
        miss = j['HYBAS_ID'].isna()
        if miss.any():
            near = gpd.sjoin_nearest(points[miss].to_crs(3857), b.to_crs(3857), how='left')
            j.loc[miss, 'HYBAS_ID'] = near[~near.index.duplicated()]['HYBAS_ID']
        out[f'basin_l{lev:02d}'] = j['HYBAS_ID'].astype('Int64').astype(str)
    return out


def nearest_distance(src, layer):
    if layer.empty:
        return pd.Series(np.nan, index=src.index)
    j = gpd.sjoin_nearest(src, layer[['geometry']], how='left', distance_col='d')
    return j.groupby(level=0)['d'].min().reindex(src.index)


def osm_context(fp):
    out = pd.DataFrame({'lake_phys': fp['lake_phys'], 'anthropogenic': False,
                        'anthropogenic_tag': None, 'dist_road_m': np.nan,
                        'dist_settlement_m': np.nan}, index=fp.index)
    for acq, sub in fp.groupby('acquisition'):
        path = OSM_DIR / f'{acq}.gpkg'
        if not path.exists():
            print(f'  WARNING: no OSM context for {acq} ({path.name} missing)')
            continue
        epsg = EXPANDED_STUDY_AREAS[acq]['epsg']
        utm = sub[['geometry']].to_crs(epsg)
        anth = gpd.read_file(path, layer='anthropogenic').to_crs(epsg)
        if not anth.empty:
            j = gpd.sjoin_nearest(utm, anth[['tag', 'geometry']], how='inner',
                                  max_distance=ANTHRO_DIST_M, distance_col='d')
            j = j.sort_values('d')
            tags = j.groupby(level=0)['tag'].agg(lambda t: ';'.join(dict.fromkeys(t)))
            out.loc[tags.index, 'anthropogenic'] = True
            out.loc[tags.index, 'anthropogenic_tag'] = tags
        for layer, col in (('roads', 'dist_road_m'), ('settlements', 'dist_settlement_m')):
            out.loc[sub.index, col] = nearest_distance(utm, gpd.read_file(path, layer=layer).to_crs(epsg))
    agg = out.groupby('lake_phys').agg(
        anthropogenic=('anthropogenic', 'any'),
        anthropogenic_tag=('anthropogenic_tag', lambda t: ';'.join(dict.fromkeys(t.dropna())) or None),
        dist_road_m=('dist_road_m', 'min'),
        dist_settlement_m=('dist_settlement_m', 'min'))
    print(f'  OSM context: {int(agg["anthropogenic"].sum())} anthropogenic of {len(agg)} lakes')
    return agg


def footprint_validity(fp):
    rows = []
    for acq in sorted(fp['acquisition'].unique()):
        sub = fp[fp['acquisition'] == acq].reset_index(drop=True)
        for sensor, root in RAW.items():
            area_dir = root / acq
            if not area_dir.exists():
                continue
            for year_dir in sorted(d for d in area_dir.iterdir() if d.name.isdigit()):
                path = year_dir / 'mndwi.tif'
                if not path.exists():
                    continue
                with rasterio.open(path) as src:
                    arr = src.read(1)
                    geoms = sub.to_crs(src.crs).geometry
                    lab = rasterize(((g, i + 1) for i, g in enumerate(geoms)),
                                    out_shape=arr.shape, transform=src.transform,
                                    fill=0, dtype='int32', all_touched=True)
                tot = np.bincount(lab.ravel(), minlength=len(sub) + 1)[1:]
                ok = np.bincount(lab.ravel(), weights=np.isfinite(arr).ravel(),
                                 minlength=len(sub) + 1)[1:]
                frac = np.where(tot > 0, ok / np.maximum(tot, 1), np.nan)
                rows.append(pd.DataFrame({'lake_phys': sub['lake_phys'], 'year': int(year_dir.name),
                                          'sensor': sensor, 'valid_frac': frac}))
        print(f'  validity {acq}: done')
    return pd.concat(rows, ignore_index=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    g = gpd.read_file(FEATURES_GPKG, columns=['lake_id', 'area_name', 'year', 'geometry'])
    print(f'  lake-year geometries: {len(g)}')

    phys = name_lakes(cluster_lakes(g))
    ext = phys.groupby('lake_phys').apply(
        lambda d: float(np.hypot(d['cx'].max() - d['cx'].min(), d['cy'].max() - d['cy'].min())),
        include_groups=False)
    print(f'  physical lakes: {phys["lake_phys"].nunique()}  '
          f'centroid spread p50={ext.median():.0f} m p99={ext.quantile(0.99):.0f} m '
          f'>1 km: {(ext > 1000).sum()}')

    geom = g.set_index('lake_id').geometry
    fp_rows = []
    for (lp, acq, epsg), d in phys.groupby(['lake_phys', 'acquisition', 'epsg']):
        parts = gpd.GeoSeries(geom.loc[d['lake_id']].values, crs=g.crs).to_crs(epsg).make_valid()
        try:
            union = parts.union_all()
        except shapely.errors.GEOSException:
            union = parts.union_all(grid_size=0.01)
        fp_rows.append({'lake_phys': lp, 'acquisition': acq,
                        'geometry': gpd.GeoSeries([union], crs=epsg).to_crs(4326).iloc[0]})
    fp = gpd.GeoDataFrame(fp_rows, crs=4326)
    fp['geometry'] = fp.geometry.make_valid()

    cent = fp.dissolve('lake_phys').geometry.representative_point()
    pts = gpd.GeoDataFrame(geometry=cent, crs=4326)
    country = countries(pts)
    lake_tab = pd.DataFrame({'lon': cent.x, 'lat': cent.y, 'country': country}).join(basins(pts))
    lake_tab = lake_tab.join(osm_context(fp))

    phys = phys.join(lake_tab, on='lake_phys')
    phys[['lake_id', 'lake_phys', 'acquisition', 'analysis_region', 'year', 'country',
          'lon', 'lat'] + [f'basin_l{lev:02d}' for lev in BASIN_LEVELS] + OSM_COLS].to_csv(
        MAP_FILE, index=False)
    fp = fp.join(lake_tab, on='lake_phys')
    fp['analysis_region'] = fp['acquisition'].map(analysis_region)
    fp.to_file(FOOTPRINTS, driver='GPKG')
    print(lake_tab['country'].value_counts().to_string())

    val = footprint_validity(fp)
    det = phys[['lake_phys', 'year']].drop_duplicates().assign(detected=1)
    val = (val.groupby(['lake_phys', 'year'], as_index=False)['valid_frac'].max()
           .merge(det, on=['lake_phys', 'year'], how='left'))
    val['detected'] = val['detected'].fillna(0).astype(int)
    val['valid'] = ((val['valid_frac'] >= VALID_FRAC) | (val['detected'] == 1)).astype(int)
    val.to_csv(VALIDITY_FILE, index=False)
    print(f'  saved {MAP_FILE.name}, {FOOTPRINTS.name}, {VALIDITY_FILE.name}')


if __name__ == '__main__':
    main()
