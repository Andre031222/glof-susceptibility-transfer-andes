import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, box

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def circle(x, y, area):
    return Point(x, y).buffer(np.sqrt(area / np.pi), 64)


def test_v5_core():
    v5 = load('v5_inventory_compare')
    epsg = 32718
    domain = box(0, 0, 10000, 10000)
    ref = gpd.GeoDataFrame({'ref_id': [f'r{i}' for i in range(7)]}, geometry=[
        circle(1000, 1000, 40000), circle(3000, 1000, 15000), circle(5000, 1000, 2e5),
        circle(7000, 1000, 5000), circle(9000, 9000, 30000),
        circle(2000, 5000, 20000), circle(2300, 5000, 20000)], crs=epsg)
    ref = pd.concat([ref, gpd.GeoDataFrame({'ref_id': ['out']}, geometry=[circle(15000, 1000, 50000)],
                                           crs=epsg)], ignore_index=True)
    det = gpd.GeoDataFrame({'lake_id': [f'd{i}' for i in range(5)]}, geometry=[
        circle(1000, 1000, 48000), circle(5000, 1000, 1.8e5), circle(2150, 5000, 90000),
        circle(8000, 8000, 20000), circle(15000, 1000, 50000)], crs=epsg)
    r, d, p = v5.compare(ref, det, domain)
    m = v5.metrics(r, d, p)
    assert m['n_ref'] == 6, m
    assert m['n_ref_matched'] == 4, m
    assert m['n_det'] == 4, m
    assert m['n_det_matched'] == 3, m
    assert abs(m['recall'] - 4 / 6) < 1e-9 and abs(m['precision'] - 3 / 4) < 1e-9
    assert m['n_pairs'] == 3, p
    merged = p[p['n_ref'] == 2]
    assert len(merged) == 1 and abs(merged['area_ref_m2'].iloc[0] - 40000) < 200
    rel = 100 * ((48000 - 40000) + (1.8e5 - 2e5) + (90000 - 40000)) / (40000 + 2e5 + 40000)
    assert abs(m['rel_area_bias_pct'] - rel) < 0.5, (m['rel_area_bias_pct'], rel)
    rows = v5.metric_rows(r, d, p, {'area_name': 't'})
    assert rows[0]['size_class'] == 'all' and len(rows) > 1
    print(f'[ok] v5 core: recall={m["recall"]:.3f} precision={m["precision"]:.3f} '
          f'f1={m["f1"]:.3f} rel_bias={m["rel_area_bias_pct"]:.2f}% mae={m["area_mae_m2"]:.0f} m2')


def test_v5_cli():
    from config_expanded_study_areas import EXPANDED_STUDY_AREAS
    from src.detection import RGI_PATH
    cfg = EXPANDED_STUDY_AREAS['cordillera_blanca']
    epsg = cfg['epsg']
    gl = gpd.read_file(RGI_PATH, bbox=tuple(cfg['bbox'])).to_crs(epsg)
    inner = gpd.GeoSeries([box(*cfg['bbox'])], crs=4326).to_crs(epsg).iloc[0].buffer(-3000)
    gl = gl[gl.geometry.area > 5e5]
    pts = gl.geometry.representative_point()
    pts = pts[pts.within(inner)].iloc[::7].iloc[:12].reset_index(drop=True)
    assert len(pts) == 12
    ref = [circle(p.x, p.y, 30000) for p in pts[:10]]
    det = [circle(p.x + 20, p.y, 36000) for p in pts[:8]] + [circle(p.x, p.y, 25000) for p in pts[10:12]]
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / 'data' / 'processed' / 'lakes_s2').mkdir(parents=True)
        gpd.GeoDataFrame({'lake_id': [f'cb_{i}' for i in range(len(det))], 'area_name': 'cordillera_blanca',
                          'year': 2019}, geometry=det, crs=epsg).to_file(
            tmp / 'data' / 'processed' / 'lakes_s2' / 'cordillera_blanca_2019.gpkg', driver='GPKG')
        gpd.GeoDataFrame({'name': range(len(ref))}, geometry=ref, crs=epsg).to_crs(4326).to_file(
            tmp / 'ref.gpkg', driver='GPKG')
        env = dict(os.environ, GLOF_ROOT=str(tmp))
        cmd = [sys.executable, '-I', str(HERE / 'v5_inventory_compare.py'), '--reference', 'file',
               '--ref-path', str(tmp / 'ref.gpkg'), '--ref-year', '2020', '--areas', 'cordillera_blanca',
               '--sensor', 's2', '--out-dir', str(tmp / 'models')]
        r = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        t = pd.read_csv(tmp / 'models' / 'inventory_validation.csv')
        row = t[(t.area_name == 'cordillera_blanca') & (t.size_class == 'all')].iloc[0]
        assert row.det_year == 2019 and row.ref_year == 2020
        assert row.n_ref == 10 and row.n_ref_matched == 8, row
        assert row.n_det == 10 and row.n_det_matched == 8, row
        assert abs(row.rel_area_bias_pct - 20.0) < 1.0, row
        mt = pd.read_csv(tmp / 'models' / 'inventory_validation_matches_file.csv')
        assert set(mt['kind']) == {'reference', 'detection'} and len(mt) == 20
    print(f'[ok] v5 CLI on synthetic lakes beside real RGI glaciers: recall={row.recall:.2f} '
          f'precision={row.precision:.2f} f1={row.f1:.2f} rel_bias={row.rel_area_bias_pct:.1f}%')


def synth_population(rng, n_neg=3000, n_pos=30, areas=('a1', 'a2', 'a3')):
    score = np.r_[rng.beta(1.2, 6, n_neg), rng.beta(4, 2, n_pos)]
    y = np.r_[np.zeros(n_neg, int), np.ones(n_pos, int)]
    df = pd.DataFrame({'lake_key': [f'L{i:05d}' for i in range(len(y))],
                       'area_name': rng.choice(areas, len(y)), 'lat': -10.0, 'lon': -77.0,
                       'area_m2': rng.lognormal(10.5, 1.0, len(y)), 'dist_glacier_m': 1000.0,
                       'elev_mean': 4500.0, 'oof': score, 'y': y})
    p_comm = np.clip(0.30 - 0.25 * df['oof'], 0.02, None)
    df['true_real'] = (rng.random(len(df)) >= p_comm).astype(int)
    p_hidden = np.where(df['y'] == 0, 0.02 + 0.6 * df['oof'] ** 2, 0.0)
    df['true_breach'] = ((rng.random(len(df)) < p_hidden) & (df['true_real'] == 1)).astype(int)
    return df


def make_sheet(v1, pop, rng):
    lakes = pop.copy()
    k = max(1, int(round(len(lakes) * v1.TOP_PCT / 100)))
    thr = lakes['oof'].sort_values(ascending=False).iloc[k - 1]
    lakes['in_watchlist_top2pct'] = (lakes['oof'] >= thr).astype(int)
    lakes = v1.design_sample(lakes, rng)
    idx = []
    for _, g in lakes.groupby('area_name'):
        idx.extend(rng.choice(g.index.values, size=min(v1.PER_AREA, len(g)), replace=False))
    keep = sorted(set(idx) | set(lakes.index[lakes.in_watchlist_top2pct == 1])
                  | set(lakes.index[lakes.y == 1]) | set(lakes.index[lakes.in_design_sample == 1]))
    s = lakes.loc[keep].rename(columns={'oof': 'model_score', 'y': 'known_glof_source'})
    return s


def make_results(sheet, rng, nonresponse=0.0):
    rows = []
    for _, r in sheet.iterrows():
        if rng.random() < nonresponse and r.known_glof_source == 0 and r.in_watchlist_top2pct == 0:
            continue
        real = 'Y' if r.true_real else 'N'
        breach = 'Y' if r.true_breach else ('N' if r.true_real else '')
        rows.append({'lake_key': r.lake_key, 'reviewer': 'A', 'area_name': r.area_name,
                     'area_ha': r.area_m2 / 1e4, 'model_score': r.model_score,
                     'in_watchlist_top2pct': r.in_watchlist_top2pct,
                     'known_glof_source': r.known_glof_source, 'design_stratum': r.design_stratum,
                     'inclusion_prob': r.inclusion_prob, 'in_design_sample': r.in_design_sample,
                     'is_real_glacial_lake': real, 'feature_type': '', 'outburst_evidence': breach,
                     'note': ''})
    return pd.DataFrame(rows)


def estimates(v3, sheet, res, oof=None):
    lab, _ = v3.consensus(res)
    frame = v3.frame_from(sheet, res).merge(lab, left_on='lake_key', right_index=True, how='left')
    frame, sizes = v3.assign_cells(frame)
    rows, meta = v3.design_validation(frame, sizes, oof)
    return {(r['estimand'], r['domain']): r for r in rows}, sizes, meta


def test_v3_design():
    v1 = load('v1_validation_sheet')
    v3 = load('v3_process_validation')
    rng = np.random.RandomState(7)
    pop = synth_population(rng)
    neg = pop[pop.y == 0]
    truth_comm = 1 - pop['true_real'].mean()
    truth_hidden = neg['true_breach'].mean()
    sheet = make_sheet(v1, pop, np.random.RandomState(1))
    res = make_results(sheet, rng)
    res2 = res.iloc[:40].copy()
    res2['reviewer'] = 'B'
    est, sizes, meta = estimates(v3, sheet, pd.concat([res, res2]))
    assert int(sizes.sum()) == len(pop), (sizes.sum(), len(pop))
    e = est[('commission_rate', 'overall')]
    assert abs(e['N_hat'] - len(pop)) < 1e-6
    wl_true = 1 - pop.loc[sheet.index[sheet.in_watchlist_top2pct == 1], 'true_real'].mean()
    ew = est[('commission_rate', 'watchlist_top2pct')]
    assert abs(ew['estimate'] - wl_true) < 1e-9 and ew['se'] < 1e-9

    cov = {'comm': 0, 'hidden': 0, 'p5': 0}
    draws = {'comm': [], 'hidden': [], 'p5': []}
    reps = 150
    oof_rows = pop[['lake_key', 'oof', 'y']].rename(columns={'oof': 'oof_score', 'y': 'glof'})
    with tempfile.TemporaryDirectory() as tmp:
        oof_path = Path(tmp) / 'oof_scores.csv'
        oof_rows.to_csv(oof_path, index=False)
        k5 = int(round(len(pop) * 0.05))
        top5 = pop.sort_values('oof', ascending=False).iloc[:k5]
        truth_p5 = ((top5.y == 1) | (top5.true_breach == 1)).mean()
        for b in range(reps):
            r = np.random.RandomState(100 + b)
            sh = make_sheet(v1, pop, r)
            rs = make_results(sh, r, nonresponse=0.1)
            est, _, _ = estimates(v3, sh, rs, oof_path)
            for key, tr, name in ((('commission_rate', 'overall'), truth_comm, 'comm'),
                                  (('missing_positive_rate', 'negatives'), truth_hidden, 'hidden'),
                                  (('precision_at_k_corrected', 'top_5pct'), truth_p5, 'p5')):
                x = est[key]
                cov[name] += int(abs(x['estimate'] - tr) <= 1.96 * x['se'] + 1e-12)
                draws[name].append(x['estimate'])
    rates = {k: v / reps for k, v in cov.items()}
    truths = {'comm': truth_comm, 'hidden': truth_hidden, 'p5': truth_p5}
    for k, v in draws.items():
        v = np.asarray(v)
        assert abs(v.mean() - truths[k]) < 3 * v.std() / np.sqrt(reps), (k, v.mean(), truths[k])
    assert rates['comm'] >= 0.88 and rates['hidden'] >= 0.82 and rates['p5'] >= 0.75, rates
    print(f'[ok] v3 design estimators: truth commission={truth_comm:.4f}, hidden-positive rate '
          f'among negatives={truth_hidden:.4f}, corrected precision@5%={truth_p5:.4f}; '
          f'mean estimates {dict((k, round(float(np.mean(v)), 4)) for k, v in draws.items())}; '
          f'95% CI coverage over {reps} resamples with 10% non-response: {rates}')

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        sheet.drop(columns=['true_real', 'true_breach']).to_csv(tmp / 'sheet.csv', index=False)
        res.drop(columns=[]).to_csv(tmp / 'results.csv', index=False)
        cmd = [sys.executable, '-I', str(HERE / 'v3_process_validation.py'), str(tmp / 'results.csv'),
               '--sheet', str(tmp / 'sheet.csv'), '--oof', str(tmp / 'none.csv'),
               '--out-dir', str(tmp / 'models')]
        r = subprocess.run(cmd, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr
        js = json.loads((tmp / 'models' / 'design_validation.json').read_text())
        csv = pd.read_csv(tmp / 'models' / 'design_validation.csv')
        assert js['population_N'] > 0 and len(csv) == len(js['estimates'])
        assert (tmp / 'models' / 'inventory_validation_summary.csv').exists()
        assert {'precision_at_k_corrected', 'missing_positive_rate', 'commission_rate'} <= set(csv.estimand)
    print(f'[ok] v3 CLI wrote design_validation.json/.csv ({len(csv)} estimates, '
          f'threshold source: {js["threshold_source"]})')


def test_v2_template():
    v2 = load('v2_build_review_app')
    js = v2.HTML_TEMPLATE.split('<script>')[1].split('</script>')[0].replace('__DATA__', '[]')
    with tempfile.NamedTemporaryFile('w', suffix='.js', delete=False) as f:
        f.write(js)
    r = subprocess.run(['node', '--check', f.name], capture_output=True, text=True)
    os.unlink(f.name)
    assert r.returncode == 0, r.stderr
    assert 'outburst_evidence' in js and 'inclusion_prob' in js
    print('[ok] v2 template script parses and exports design fields and outburst_evidence')


if __name__ == '__main__':
    test_v5_core()
    test_v5_cli()
    test_v3_design()
    test_v2_template()
    print('all self-tests passed')
