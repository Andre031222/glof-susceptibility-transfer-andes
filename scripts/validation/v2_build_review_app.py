import csv
import json
import math
import ssl
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHEET = PROJECT_ROOT / 'docs' / 'inventory_validation_sheet.csv'
APP_DIR = PROJECT_ROOT / 'docs' / 'validation_app'
THUMB_DIR = APP_DIR / 'thumbs'
SIZE = 340
EXPORT = ("https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/"
          "MapServer/export")
CTX = ssl.create_default_context()


def bbox_for(lat, lon, area_m2):
    r = max(350.0, 2.2 * math.sqrt(max(area_m2, 1.0) / math.pi))
    dlat = r / 111000.0
    dlon = r / (111000.0 * max(0.2, math.cos(math.radians(lat))))
    return lon - dlon, lat - dlat, lon + dlon, lat + dlat


def fetch(job):
    idx, lat, lon, area = job
    out = THUMB_DIR / f'{idx}.jpg'
    if out.exists() and out.stat().st_size > 1000:
        return idx, True
    x0, y0, x1, y1 = bbox_for(lat, lon, area)
    url = (f"{EXPORT}?bbox={x0},{y0},{x1},{y1}&bboxSR=4326&imageSR=4326"
           f"&size={SIZE},{SIZE}&format=jpg&f=image")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        data = urllib.request.urlopen(req, timeout=30, context=CTX).read()
        if data[:3] == b'\xff\xd8\xff':
            out.write_bytes(data)
            return idx, True
    except Exception:
        pass
    return idx, False


def main():
    THUMB_DIR.mkdir(parents=True, exist_ok=True)
    def num(v, d=0.0):
        try:
            return float(v)
        except (ValueError, TypeError):
            return d

    rows = list(csv.DictReader(open(SHEET)))
    for i, r in enumerate(rows):
        r['idx'] = i
        r['lat'] = num(r['lat']); r['lon'] = num(r['lon'])
        r['area_m2'] = num(r['area_m2'])
        r['area_ha'] = round(r['area_m2'] / 1e4, 2)
        r['model_score'] = round(num(r['model_score']), 3)
        r['dist_glacier_m'] = round(num(r['dist_glacier_m']))
        r['elev_mean'] = round(num(r['elev_mean']))
        r['design_stratum'] = int(num(r.get('design_stratum'), 0)) or ''
        r['inclusion_prob'] = num(r.get('inclusion_prob'), None)
        r['in_design_sample'] = int(num(r.get('in_design_sample'), 0))
    rows = [r for r in rows if r['lat'] and r['lon']]

    rows.sort(key=lambda r: (r['area_name'], -int(r['in_watchlist_top2pct']),
                             -r['model_score']))

    jobs = [(r['idx'], r['lat'], r['lon'], r['area_m2']) for r in rows]
    print(f'Downloading {len(jobs)} thumbnails -> {THUMB_DIR} ...')
    ok = 0
    with ThreadPoolExecutor(max_workers=8) as ex:
        for n, (idx, good) in enumerate(ex.map(fetch, jobs), 1):
            ok += good
            if n % 100 == 0:
                print(f'  {n}/{len(jobs)} ({ok} ok)')
    print(f'  done: {ok}/{len(jobs)} thumbnails')

    data = [{
        'idx': r['idx'], 'lake_key': r['lake_key'], 'area': r['area_name'],
        'area_ha': r['area_ha'], 'score': r['model_score'],
        'wl': int(r['in_watchlist_top2pct']), 'known': int(r['known_glof_source']),
        'dist': r['dist_glacier_m'], 'elev': r['elev_mean'],
        'lat': round(r['lat'], 5), 'lon': round(r['lon'], 5),
        'stratum': r['design_stratum'],
        'pi': None if r['inclusion_prob'] is None else round(r['inclusion_prob'], 8),
        'design': r['in_design_sample'],
    } for r in rows]

    html = HTML_TEMPLATE.replace('__DATA__', json.dumps(data))
    (APP_DIR / 'index.html').write_text(html)
    print('wrote', APP_DIR / 'index.html')
    print('\nAbrir en el navegador:')
    print(f'  file://{APP_DIR / "index.html"}')


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8">
<title>GLOF inventory validation</title>
<style>
 body{font-family:system-ui,Arial,sans-serif;margin:0;background:#0f1419;color:#e6e6e6}
 header{position:sticky;top:0;background:#1a2230;padding:10px 16px;z-index:10;
   border-bottom:1px solid #2c3a4f;display:flex;gap:14px;align-items:center;flex-wrap:wrap}
 header h1{font-size:15px;margin:0;font-weight:600}
 .stat{font-size:13px;color:#9db4d0}
 button,select{font-size:13px;padding:5px 9px;border-radius:6px;border:1px solid #3a4a63;
   background:#222d3d;color:#e6e6e6;cursor:pointer}
 button.primary{background:#2563eb;border-color:#2563eb}
 .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px;padding:14px}
 .card{background:#172030;border:1px solid #283545;border-radius:10px;overflow:hidden;
   display:flex;flex-direction:column}
 .card.wl{border-color:#e0a000}
 .card.done{outline:2px solid #16a34a}
 .imgwrap{position:relative;aspect-ratio:1;background:#0a0e14}
 .imgwrap img{width:100%;height:100%;object-fit:cover;display:block}
 .cross{position:absolute;top:50%;left:50%;width:26px;height:26px;transform:translate(-50%,-50%);
   pointer-events:none}
 .cross:before,.cross:after{content:"";position:absolute;background:#ff3b3b}
 .cross:before{left:50%;top:0;width:2px;height:100%;transform:translateX(-50%)}
 .cross:after{top:50%;left:0;height:2px;width:100%;transform:translateY(-50%)}
 .ring{position:absolute;top:50%;left:50%;width:46px;height:46px;border:2px solid rgba(255,59,59,.6);
   border-radius:50%;transform:translate(-50%,-50%);pointer-events:none}
 .meta{padding:7px 9px;font-size:12px;line-height:1.5}
 .badge{display:inline-block;font-size:10px;padding:1px 6px;border-radius:4px;margin-left:4px}
 .b-wl{background:#e0a000;color:#000}.b-known{background:#dc2626;color:#fff}
 .ctrl{padding:7px 9px;border-top:1px solid #283545;display:flex;flex-direction:column;gap:5px}
 .yn{display:flex;gap:6px}.yn button{flex:1}
 .yn button.on-y{background:#16a34a;border-color:#16a34a}
 .yn button.on-n{background:#dc2626;border-color:#dc2626}
 a{color:#6db3ff}
</style></head><body>
<header>
 <h1>GLOF inventory validation</h1>
 <span class="stat" id="prog"></span>
 <label class="stat">Filtro:
  <select id="filter">
   <option value="all">Todos</option>
   <option value="wl">Solo watch-list</option>
   <option value="todo">Pendientes</option>
   <option value="design">Muestra probabilistica (diseno + watch-list + GLOF)</option>
   <option value="off">Solo off-context (lejos hielo / baja elev)</option>
  </select></label>
 <button class="primary" onclick="exportCSV()">Descargar CSV de resultados</button>
 <span class="stat">Guardado automatico en este navegador</span>
</header>
<div class="grid" id="grid"></div>
<script>
const DATA = __DATA__;
const TYPES = ["","glacial","proglacial","moraine_dammed","bedrock","reservoir",
  "river","wetland","cloud_shadow","snow_ice","other"];
const KEY = "glof_val_v1";
let marks = JSON.parse(localStorage.getItem(KEY) || "{}");
function save(){localStorage.setItem(KEY, JSON.stringify(marks));upProg();}
function upProg(){
  const n=DATA.length, d=Object.keys(marks).filter(k=>marks[k]&&marks[k].real).length;
  const wl=DATA.filter(x=>x.wl).length;
  const wld=DATA.filter(x=>x.wl&&marks[x.idx]&&marks[x.idx].real).length;
  document.getElementById('prog').textContent=
    `${d}/${n} revisados  ·  watch-list ${wld}/${wl}`;
}
function isOff(x){return x.dist>9830 || x.elev<2899;}
function render(){
  const f=document.getElementById('filter').value;
  const g=document.getElementById('grid');g.innerHTML='';
  DATA.filter(x=>{
    if(f==='wl')return x.wl;
    if(f==='todo')return !(marks[x.idx]&&marks[x.idx].real);
    if(f==='off')return isOff(x);
    if(f==='design')return x.design||x.known||x.wl;
    return true;
  }).forEach(x=>{
    const m=marks[x.idx]||{};
    const c=document.createElement('div');
    c.className='card'+(x.wl?' wl':'')+(m.real?' done':'');
    c.innerHTML=`
     <div class="imgwrap">
       <img loading="lazy" src="thumbs/${x.idx}.jpg" alt="">
       <div class="ring"></div><div class="cross"></div>
     </div>
     <div class="meta">
       <b>${x.area}</b>${x.wl?'<span class="badge b-wl">WATCH</span>':''}
       ${x.known?'<span class="badge b-known">GLOF</span>':''}<br>
       ${x.area_ha} ha · score ${x.score} · ${x.dist} m al hielo · ${x.elev} m${isOff(x)?' · off-context':''}<br>
       <a href="https://earth.google.com/web/@${x.lat},${x.lon},1000a,2500d,35y,0h,0t,0r" target="_blank">Google Earth</a>
     </div>
     <div class="ctrl">
       <div class="yn">
         <button class="${m.real==='Y'?'on-y':''}" onclick="setReal(${x.idx},'Y')">Lago glaciar</button>
         <button class="${m.real==='N'?'on-n':''}" onclick="setReal(${x.idx},'N')">No es</button>
       </div>
       <select onchange="setType(${x.idx},this.value)">
         ${TYPES.map(t=>`<option value="${t}" ${m.type===t?'selected':''}>${t||'-- tipo --'}</option>`).join('')}
       </select>
       <select onchange="setBreach(${x.idx},this.value)" title="Evidencia geomorfica de vaciamiento pasado (morrena rota, canal de brecha, abanico de descarga)">
         <option value="" ${!m.breach?'selected':''}>-- evidencia de GLOF pasado --</option>
         <option value="Y" ${m.breach==='Y'?'selected':''}>Si: morrena rota / brecha</option>
         <option value="N" ${m.breach==='N'?'selected':''}>No se observa</option>
       </select>
       <input placeholder="nota (opcional)" value="${(m.note||'').replace(/"/g,'&quot;')}"
         oninput="setNote(${x.idx},this.value)">
     </div>`;
    g.appendChild(c);
  });
}
function setReal(i,v){marks[i]=marks[i]||{};marks[i].real=v;save();render();}
function setType(i,v){marks[i]=marks[i]||{};marks[i].type=v;save();}
function setBreach(i,v){marks[i]=marks[i]||{};marks[i].breach=v;save();}
function setNote(i,v){marks[i]=marks[i]||{};marks[i].note=v;save();}
function exportCSV(){
  const head=["lake_key","area_name","lat","lon","area_ha","model_score",
    "in_watchlist_top2pct","known_glof_source","design_stratum","inclusion_prob",
    "in_design_sample","is_real_glacial_lake","feature_type","outburst_evidence","note"];
  const lines=[head.join(",")];
  DATA.forEach(x=>{const m=marks[x.idx]||{};
    lines.push([x.lake_key,x.area,x.lat,x.lon,x.area_ha,x.score,x.wl,x.known,
      x.stratum,x.pi==null?"":x.pi,x.design,
      m.real||"",m.type||"",m.breach||"",'"'+(m.note||"").replace(/"/g,'""')+'"'].join(","));});
  const blob=new Blob([lines.join("\n")],{type:"text/csv"});
  const a=document.createElement("a");a.href=URL.createObjectURL(blob);
  a.download="inventory_validation_results.csv";a.click();
}
document.getElementById('filter').onchange=render;
render();upProg();
</script></body></html>"""


if __name__ == '__main__':
    main()
