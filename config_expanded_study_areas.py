EXPANDED_STUDY_AREAS = {
    'cordillera_blanca': {
        'description': 'Cordillera Blanca, Ancash, Peru — largest tropical glacier concentration',
        'bbox': [-77.8, -9.8, -77.1, -8.8],
        'epsg': 32718,
        'elev_min_m': 3000,
        'elev_max_m': 6500,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 3083,
        'glof_events_documented': 10,
        'status': 'COMPLETED',
        'priority': 'CRITICAL',
        'notes': '9 of 16 matched GLOF labels; highest GLOF density in dataset.',
    },

    'cordillera_vilcanota': {
        'description': 'Cordillera Vilcanota, Cusco, Peru',
        'bbox': [-71.2, -14.172, -70.4, -13.2],
        'epsg': 32719,
        'elev_min_m': 3000,
        'elev_max_m': 6100,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 2770,
        'glof_events_documented': 0,
        'status': 'COMPLETED',
        'priority': 'HIGH',
        'notes': '5 inventory events but 0 matched (coordinates correspond to downstream zones).',
    },

    'cordillera_central': {
        'description': 'Cordillera Central, Junin-Lima, Peru',
        'bbox': [-76.474, -11.805, -75.8, -11.0],
        'epsg': 32718,
        'elev_min_m': 3000,
        'elev_max_m': 5100,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 2239,
        'glof_events_documented': 0,
        'status': 'COMPLETED',
        'priority': 'HIGH',
        'notes': '4 inventory events, 0 matched. Lakes well-detected; event coords unreliable.',
    },

    'chile_andes_centrales': {
        'description': 'Andes Centrales, Santiago Metropolitan Region, Chile',
        'bbox': [-70.304, -34.222, -69.5, -33.0],
        'epsg': 32719,
        'elev_min_m': 2000,
        'elev_max_m': 6600,
        'dry_season_months': [12, 1, 2],
        'max_cloud_cover': 20,
        'lakes_estimated': 1589,
        'glof_events_documented': 0,
        'status': 'COMPLETED',
        'priority': 'MEDIUM',
        'notes': 'Null-class domain (0 matched GLOF labels). FPR@0.304=29.6%. '
                 'Mediterranean seasonality differs from tropical Andes.',
    },

    'cordillera_raura': {
        'description': 'Cordillera Raura, Lima-Huanuco, Peru',
        'bbox': [-76.85, -10.589, -76.53, -10.206],
        'epsg': 32718,
        'elev_min_m': 3700,
        'elev_max_m': 5700,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 1361,
        'glof_events_documented': 2,
        'status': 'COMPLETED',
        'priority': 'HIGH',
        'notes': '2 matched positives; highest high-risk fraction (55.7%).',
    },

    'cordillera_urubamba': {
        'description': 'Cordillera Urubamba, Cusco, Peru',
        'bbox': [-72.624, -13.489, -71.8, -13.0],
        'epsg': 32719,
        'elev_min_m': 2300,
        'elev_max_m': 5800,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 430,
        'glof_events_documented': 0,
        'status': 'COMPLETED',
        'priority': 'MEDIUM',
        'notes': '3 inventory events, 0 matched (17-40 km away from detected lakes).',
    },

    'cordillera_huanzo': {
        'description': 'Cordillera Huanzo, Apurimac-Arequipa, Peru',
        'bbox': [-73.658, -15.3, -72.7, -14.645],
        'epsg': 32718,
        'elev_min_m': 1800,
        'elev_max_m': 5300,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 385,
        'glof_events_documented': 0,
        'status': 'COMPLETED',
        'priority': 'MEDIUM',
        'notes': '4 inventory events, 0 matched (31-57 km away). Remote area.',
    },

    'cordillera_huayhuash': {
        'description': 'Cordillera Huayhuash, Ancash-Lima-Huanuco, Peru',
        'bbox': [-77.057, -10.35, -76.75, -10.05],
        'epsg': 32718,
        'elev_min_m': 3500,
        'elev_max_m': 6500,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 278,
        'glof_events_documented': 4,
        'status': 'COMPLETED',
        'priority': 'CRITICAL',
        'notes': '3 matched positives from v1 inventory; adding 2023 Lago Rasac event (Emmer2025).',
    },

    'ecuador_antisana': {
        'description': 'Antisana Ecological Reserve, Napo-Pichincha, Ecuador',
        'bbox': [-78.3, -0.6, -78.0, -0.3],
        'epsg': 32717,
        'elev_min_m': 2100,
        'elev_max_m': 5800,
        'dry_season_months': [12, 1, 2],
        'max_cloud_cover': 40,
        'lakes_estimated': 273,
        'glof_events_documented': 1,
        'status': 'COMPLETED',
        'priority': 'HIGH',
        'notes': 'Highest LOCO AUC (0.993); 1 matched positive. Higher cloud cover accepted.',
    },

    'bolivia_cordillera_real': {
        'description': 'Cordillera Real, La Paz, Bolivia',
        'bbox': [-68.35, -16.65, -67.7, -15.75],
        'epsg': 32719,
        'elev_min_m': 3500,
        'elev_max_m': 6400,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 180,
        'glof_events_documented': 1,
        'status': 'COMPLETED',
        'priority': 'HIGH',
        'notes': '7 inventory events, 1 matched (Tuni-Condoriri). Gap bridged by new Apolobamba area.',
    },

    'patagonia_norte': {
        'description': 'NPI Cachet2/Colonia GLOF cluster, Aysen, Chile (~47.0-47.9S)',
        'bbox': [-74.105, -47.9, -72.5, -46.345],
        'epsg': 32718,
        'elev_min_m': 0,
        'elev_max_m': 3500,
        'dry_season_months': [12, 1, 2],
        'max_cloud_cover': 50,
        'lakes_estimated': 400,
        'glof_events_documented': 3,
        'status': 'PENDING',
        'priority': 'HIGH',
        'notes': (
            'Northern Patagonian Icefield (NPI). Moraine-dammed and ice-dammed lakes. '
            'GLOFs documented by IribarrenAnacona et al. (2014, 2015). '
            'Glaciers: San Rafael, San Quintin, Exploradores, Steffen, Hudson. '
            'BBOX CORRECTED 2026-05-06: narrowed to -44.5 to -48.5 S (NPI glacial zone). '
            'Previous download got data only at lat~-40S (outside glacial area). '
            'max_cloud_cover raised to 50 (was 30) for maritime Patagonia. '
            'Run NB20 again with updated config, then NB11-NB12 for this area only.'
        ),
    },

    'patagonia_sur': {
        'description': 'S2 tile T18GXN — Cachet 2 sector, lat -47.05 to -47.5°S, lon -72.5 to -73.4°W',
        'bbox': [-73.4, -47.5, -72.5, -47.05],
        'epsg': 32718,
        'elev_min_m': 0,
        'elev_max_m': 3200,
        'dry_season_months': [5, 6, 7, 8, 9, 10],
        'max_cloud_cover': 70,
        'lakes_estimated': 40,
        'glof_events_documented': 5,
        'status': 'PENDING',
        'priority': 'CRITICAL',
        'notes': (
            'BBOX CORRECTED 2026-06-21 (v3): verified via Planetary Computer STAC.\n'
            'Correct S2 tile for Cachet 2 is T18GXN (lon -73.686 to -72.192, lat -47.933 to -46.920). '
            'T18GXP (adjacent tile, cloud=1.7%) has southern boundary -47.034 and does NOT reach '
            'Cachet 2 at -47.183. Bbox must be entirely south of -47.034 to avoid T18GXP being '
            'selected. New bbox [-73.4,-47.5,-72.5,-47.05] ensures only T18GXN tiles are found.\n'
            'SEASON: months=[5,6,7,8,9,10] (JJA austral winter). '
            'Cachet 2 fills May-Nov, drains Nov-Dec.\n'
            'TARGET: Lago Cachet 2 (-47.183,-73.017) — 4 S2-era GLOFs (2017-2020) -> expected +4 n+.\n'
            'Pipeline: delete data/raw/sentinel2/patagonia_sur/, '
            'data/raw/dem/patagonia_sur/*.tif, data/interim/terrain/patagonia_sur/*.tif, '
            'data/interim/dem/patagonia_sur_dem_utm.tif — ALL DONE 2026-06-21.\n'
            'RUN: python scripts/download_patagonia_sur_jja.py  ->  NB11  ->  '
            'python scripts/run_lake_detection.py patagonia_sur  ->  NB13  ->  NB14\n'
            'Pipeline: DELETE data/raw/sentinel2/patagonia_sur/ and '
            'data/processed/lakes/patagonia_sur_*.gpkg, then run NB21 -> NB12 -> NB13 -> NB14.'
        ),
    },

    'bolivia_norte': {
        'description': 'Northern Cordillera Real, La Paz, Bolivia — Zongo Valley and northern sectors',
        'bbox': [-68.608, -15.889, -67.5, -14.3],
        'epsg': 32719,
        'elev_min_m': 3500,
        'elev_max_m': 6500,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 120,
        'glof_events_documented': 5,
        'status': 'PENDING',
        'priority': 'HIGH',
        'notes': (
            'Northern extension of Bolivia Cordillera Real. Covers Zongo Valley '
            '(Glaciar Zongo GLOF 2009), Laguna Khara Khota, Glaciar Illampu, '
            'and Keara sectors. Events are 8-34km from lakes in the southern bbox. '
            'Khara_Khota (2016) nearest lake = 8,488m -> should match with 10km buffer '
            'if lake is detected in this new area or corrected coordinate is used. '
            'Zongo_1 (2009) nearest lake = 13,360m -> possible with 14km buffer. '
            'Run new download notebook for this area, then NB11-NB12, then NB14 '
            'with graduated buffer (pre-2017: 10km, post-2017: 5km).'
        ),
    },

    'apolobamba': {
        'description': 'Cordillera Apolobamba, Puno (Peru) / La Paz (Bolivia), ~13.8-15.2°S',
        'bbox': [-69.6, -15.652, -68.3, -13.8],
        'epsg': 32719,
        'elev_min_m': 3800,
        'elev_max_m': 6100,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 350,
        'glof_events_documented': 3,
        'status': 'PENDING',
        'priority': 'HIGH',
        'notes': (
            'Fills geographic gap between Bolivia Cordillera Real and Peru Vilcanota. '
            'Transboundary area Peru-Bolivia. Several proglacial lakes documented '
            'in the INAIGEM inventory and regional literature. '
            'Similar climate to Cordillera Real: strong dry season Jun-Aug. '
            'Run NB22 first, then follow standard pipeline.'
        ),
    },

    'carabaya': {
        'description': 'Cordillera Carabaya, Puno, Peru — eastern extension of Vilcanota system',
        'bbox': [-70.5, -15.0, -69.5, -13.5],
        'epsg': 32719,
        'elev_min_m': 3500,
        'elev_max_m': 5800,
        'dry_season_months': [6, 7, 8],
        'max_cloud_cover': 15,
        'lakes_estimated': 500,
        'glof_events_documented': 2,
        'status': 'PENDING',
        'priority': 'MEDIUM',
        'notes': (
            'Eastern cordillera of Puno region; glaciers drain toward Amazon basin. '
            'Less studied than Vilcanota but shares similar glacier characteristics. '
            'Partial coverage in existing INAIGEM records. '
            'Connects Vilcanota with Apolobamba spatially.'
        ),
    },
}


EXCLUDED_AREAS = {
    'cordillera_huanzo': 'RGI 7.0 records 0 glaciers / 0 km2 of ice within the '
                         'bbox; the range is deglaciated, so no lake can meet '
                         'the 5 km glacier-proximity criterion',
    'patagonia_sur': 'bbox lies inside patagonia_norte; its May-October season '
                     'adds winter snow and lake ice to the same analysis region',
}


def active_areas():
    return [a for a in EXPANDED_STUDY_AREAS if a not in EXCLUDED_AREAS]


ANALYSIS_REGIONS = {
    'patagonia_norte': 'patagonia_npi',
    'patagonia_sur': 'patagonia_npi',
}

ANALYSIS_REGION_LABEL = {
    'patagonia_npi': 'Northern Patagonian Icefield',
}


def analysis_region(area_name: str) -> str:
    return ANALYSIS_REGIONS.get(area_name, area_name)


def analysis_region_bbox(region: str) -> list:
    members = [cfg['bbox'] for name, cfg in EXPANDED_STUDY_AREAS.items()
               if analysis_region(name) == region]
    if not members:
        raise KeyError(f"Unknown analysis region '{region}'")
    return [min(b[0] for b in members), min(b[1] for b in members),
            max(b[2] for b in members), max(b[3] for b in members)]


def analysis_region_epsg(region: str) -> int:
    for name, cfg in EXPANDED_STUDY_AREAS.items():
        if analysis_region(name) == region:
            return cfg['epsg']
    raise KeyError(f"Unknown analysis region '{region}'")


def list_analysis_regions() -> list:
    seen = []
    for name in EXPANDED_STUDY_AREAS:
        r = analysis_region(name)
        if r not in seen:
            seen.append(r)
    return seen


def get_area(name: str) -> dict:
    if name not in EXPANDED_STUDY_AREAS:
        available = sorted(EXPANDED_STUDY_AREAS.keys())
        raise KeyError(
            f"Area '{name}' not found. Available: {available}"
        )
    return EXPANDED_STUDY_AREAS[name]


def list_areas(status: str = None) -> list:
    areas = EXPANDED_STUDY_AREAS.items()
    if status:
        areas = [(k, v) for k, v in areas if v['status'] == status]
    return [k for k, _ in areas]


def pending_areas() -> list:
    return list_areas('PENDING')


def completed_areas() -> list:
    return list_areas('COMPLETED')


if __name__ == '__main__':
    print(f"Total study areas : {len(EXPANDED_STUDY_AREAS)}")
    print(f"Completed         : {len(completed_areas())}")
    print(f"Pending download  : {len(pending_areas())}")
    print()
    for name, cfg in EXPANDED_STUDY_AREAS.items():
        lakes = cfg['lakes_estimated']
        glofs = cfg['glof_events_documented']
        status = cfg['status']
        print(f"  {status:12s}  {name:30s}  ~{lakes:5d} lakes  {glofs:2d} GLOFs documented")
