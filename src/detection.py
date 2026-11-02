import calendar
import warnings
from pathlib import Path

import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import shapes, rasterize
from rasterio.warp import reproject, Resampling
from shapely.geometry import shape
from skimage.measure import label
from skimage.morphology import remove_small_objects, remove_small_holes
from scipy import ndimage

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MIN_AREA_M2 = 9000.0
MAX_AREA_M2 = 50000000.0
MNDWI_THRESH = 0.0
NDWI_THRESH = 0.1
NIR_MAX = 0.15
SCL_WATER_NIR_MAX = 0.05
MAX_LAKE_SLOPE_DEG = 15.0
GLACIER_BUFFER_M = 5000.0

RGI_PATH = PROJECT_ROOT / 'data' / 'raw' / 'rgi7_andes.gpkg'

_glacier_cache = {}
_bbox_cache = {}


def effective_bbox(area_name, bbox, epsg):
    key = (area_name, tuple(bbox), epsg)
    if key in _bbox_cache:
        return _bbox_cache[key]

    result = list(bbox)
    if RGI_PATH.exists():
        pad = GLACIER_BUFFER_M / 111000.0 * 1.5
        read_bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
        glaciers = gpd.read_file(RGI_PATH, bbox=read_bbox)
        if not glaciers.empty:
            buf = glaciers.to_crs(epsg).geometry.buffer(GLACIER_BUFFER_M)
            ice = gpd.GeoSeries([buf.union_all()], crs=epsg).to_crs(4326)
            x0, y0, x1, y1 = ice.total_bounds
            result = [max(bbox[0], float(x0)), max(bbox[1], float(y0)),
                      min(bbox[2], float(x1)), min(bbox[3], float(y1))]
            if result[0] >= result[2] or result[1] >= result[3]:
                result = list(bbox)

    _bbox_cache[key] = result
    return result


def glacier_buffer(area_name, bbox, crs):
    key = (area_name, str(crs))
    if key in _glacier_cache:
        return _glacier_cache[key]

    if not RGI_PATH.exists():
        warnings.warn(f'RGI not found: {RGI_PATH}')
        _glacier_cache[key] = None
        return None

    pad = GLACIER_BUFFER_M / 111000.0 * 1.5
    read_bbox = (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad)
    glaciers = gpd.read_file(RGI_PATH, bbox=read_bbox)
    if glaciers.empty:
        warnings.warn(f'{area_name}: no RGI glaciers within bbox')
        _glacier_cache[key] = None
        return None

    geom = glaciers.to_crs(crs).geometry
    geom = geom[geom.notna() & ~geom.is_empty].make_valid()
    geom = geom[geom.is_valid & ~geom.is_empty]
    if geom.empty:
        _glacier_cache[key] = None
        return None

    buffered = geom.buffer(GLACIER_BUFFER_M)
    buffered = buffered[buffered.is_valid & ~buffered.is_empty]
    result = None if buffered.empty else buffered.union_all()
    _glacier_cache[key] = result
    return result


def reproject_dem(dem_path, shape_hw, transform, crs):
    if dem_path is None or not Path(dem_path).exists():
        return None
    out = np.full(shape_hw, np.nan, dtype=np.float32)
    with rasterio.open(dem_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=out,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=crs,
            resampling=Resampling.bilinear,
            dst_nodata=np.nan,
        )
    return out


def season_range(year, months):
    months = list(months)
    wraps = any(b < a for a, b in zip(months, months[1:]))
    if wraps:
        start_y, start_m = year - 1, months[0]
    else:
        start_y, start_m = year, min(months)
    end_m = months[-1] if wraps else max(months)
    last_day = calendar.monthrange(year, end_m)[1]
    return f'{start_y}-{start_m:02d}-01/{year}-{end_m:02d}-{last_day:02d}'


def water_mask(mndwi, ndwi, nir, scl_water=None):
    if ndwi is None or np.all(np.isnan(ndwi)):
        raise ValueError('NDWI is required to separate water from snow and ice')
    if nir is None or np.all(np.isnan(nir)):
        raise ValueError('NIR reflectance is required to separate water from snow')
    mask = (mndwi > MNDWI_THRESH) & (ndwi > NDWI_THRESH) & (nir < NIR_MAX)
    if scl_water is not None:
        mask |= (scl_water >= 0.5) & (nir < SCL_WATER_NIR_MAX)
    return mask


def drop_steep(mask, slope):
    lab = label(mask, connectivity=2)
    if lab.max() == 0:
        return mask
    idx = np.arange(1, lab.max() + 1)
    ok = np.isfinite(slope)
    total = ndimage.sum(np.where(ok, slope, 0.0), labels=lab, index=idx)
    count = ndimage.sum(ok, labels=lab, index=idx)
    mean = np.where(count > 0, total / np.maximum(count, 1), np.inf)
    keep = np.r_[False, mean <= MAX_LAKE_SLOPE_DEG]
    return keep[lab]


def clean_mask(mask, res_m):
    min_px = max(1, int(np.ceil(MIN_AREA_M2 / (res_m ** 2))))
    m = remove_small_holes(mask, max_size=min_px - 1)
    return remove_small_objects(m, max_size=min_px - 1)


def zonal_elevation(gdf, dem, shape_hw, transform):
    lab = rasterize(((g, i + 1) for i, g in enumerate(gdf.geometry)), out_shape=shape_hw,
                    transform=transform, fill=0, dtype='int32')
    ok = np.isfinite(dem)
    lab = np.where(ok, lab, 0)
    idx = np.arange(1, len(gdf) + 1)
    n = ndimage.sum(np.ones_like(lab), labels=lab, index=idx)
    vals = np.where(ok, dem, 0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        stats = {
            'elev_mean': ndimage.mean(vals, labels=lab, index=idx),
            'elev_min': ndimage.minimum(vals, labels=lab, index=idx),
            'elev_max': ndimage.maximum(vals, labels=lab, index=idx),
            'elev_std': ndimage.standard_deviation(vals, labels=lab, index=idx),
        }
    return gpd.GeoDataFrame({k: np.where(n > 0, np.asarray(v, dtype=float), np.nan)
                             for k, v in stats.items()})


def detect_lakes(mndwi, transform, crs, area_name, cfg, year, source,
                 scene_date, ndwi=None, nir=None, dem_path=None, slope_path=None,
                 satellite=None, scl_water=None):
    empty = gpd.GeoDataFrame()
    stats = {'water_px': 0, 'polygons': 0, 'after_area': 0,
             'after_elev': 0, 'after_slope': 0, 'after_glacier': 0}

    res_m = abs(transform.a)
    mask = water_mask(mndwi, ndwi, nir, scl_water)
    stats['water_px'] = int(mask.sum())

    dem = reproject_dem(dem_path, mask.shape, transform, crs)
    if dem is not None:
        in_range = (dem >= cfg['elev_min_m']) & (dem <= cfg['elev_max_m'])
        mask = mask & (in_range | np.isnan(dem))
    elif dem_path is not None:
        warnings.warn(f'{area_name}/{year}: DEM missing, elevation filter skipped')
    stats['after_elev'] = int(mask.sum())

    mask = clean_mask(mask, res_m)
    slope = reproject_dem(slope_path, mask.shape, transform, crs)
    if slope is not None:
        mask = drop_steep(mask, slope)
    elif slope_path is not None:
        warnings.warn(f'{area_name}/{year}: slope missing, steep-surface filter skipped')
    stats['after_slope'] = int(mask.sum())
    if not mask.any():
        return empty, stats

    polys = [shape(g) for g, v in shapes(mask.astype(np.uint8), transform=transform)
             if v == 1]
    if not polys:
        return empty, stats

    gdf = gpd.GeoDataFrame({'geometry': polys}, crs=crs)
    gdf['area_m2'] = gdf.geometry.area
    stats['polygons'] = len(gdf)
    gdf = gdf[(gdf['area_m2'] >= MIN_AREA_M2) &
              (gdf['area_m2'] <= MAX_AREA_M2)].reset_index(drop=True)
    stats['after_area'] = len(gdf)

    buf = glacier_buffer(area_name, cfg['bbox'], crs)
    if buf is not None and not gdf.empty:
        gdf = gdf[gdf.geometry.centroid.within(buf)].reset_index(drop=True)
    stats['after_glacier'] = len(gdf)

    if gdf.empty:
        return gdf, stats

    if dem is not None:
        gdf = gdf.join(zonal_elevation(gdf, dem, mask.shape, transform))

    gdf['area_name'] = area_name
    gdf['year'] = year
    gdf['scene_date'] = scene_date
    gdf['source'] = source
    if satellite is not None:
        gdf['satellite'] = satellite
    gdf['lake_id'] = [f'{area_name}_{year}_{i:04d}' for i in range(len(gdf))]

    return gdf, stats
