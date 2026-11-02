const TYPES = ['', 'glacial', 'proglacial', 'moraine_dammed', 'bedrock', 'reservoir',
  'river', 'wetland', 'cloud_shadow', 'snow_ice', 'other'];
let reviewer = localStorage.getItem('glof_reviewer') || '';
let filter = 'watchlist';

const $ = s => document.querySelector(s);
const api = (p, o) => fetch('/api' + p, o).then(r => r.json());

function showModal(v) { $('#modal').classList.toggle('hidden', !v); $('#modal').classList.toggle('flex', v); }

function setReviewer(name) {
  reviewer = name.trim();
  localStorage.setItem('glof_reviewer', reviewer);
  $('#who').innerHTML = reviewer
    ? `<span class="w-6 h-6 rounded-full bg-brand-100 text-brand-700 grid place-items-center text-xs font-bold">${reviewer[0].toUpperCase()}</span>${reviewer}`
    : '';
  $('#loginBtn').textContent = reviewer ? 'Cambiar' : 'Identificarme';
}

async function load() {
  if (!reviewer) { showModal(true); return; }
  const [lakes, prog, st] = await Promise.all([
    api(`/lakes?reviewer=${encodeURIComponent(reviewer)}&filter=${filter}`),
    api(`/progress?reviewer=${encodeURIComponent(reviewer)}`),
    api('/stats'),
  ]);
  renderProgress(prog); renderStats(st); renderGrid(lakes);
}

function renderProgress(p) {
  $('#m-total').textContent = p.total.toLocaleString();
  $('#m-wl').textContent = p.watchlist.toLocaleString();
  $('#m-done').textContent = p.done.toLocaleString();
  const pct = p.total ? Math.round(100 * p.done / p.total) : 0;
  $('#bar').style.width = pct + '%';
  $('#barlbl').textContent = `${p.done}/${p.total} revisados · watch-list ${p.watchlist_done}/${p.watchlist}`;
}

function renderStats(s) {
  $('#m-rate').textContent = s.overall.rate_pct == null ? '—' : s.overall.rate_pct + '%';
  const card = (t, d) => `<div class="bg-white rounded-2xl border border-slate-200 shadow-card p-5">
      <div class="text-sm text-ink-500">${t}</div>
      <div class="text-3xl font-extrabold mt-1 tabular-nums ${d.rate_pct > 25 ? 'text-red-600' : 'text-emerald-600'}">${d.rate_pct == null ? '—' : d.rate_pct + '%'}</div>
      <div class="text-xs text-ink-500 mt-1">${d.commission || 0} comisión de ${d.n || 0} revisados</div></div>`;
  $('#statcards').innerHTML = card('Comisión global', s.overall)
    + card('Comisión en watch-list', s.watchlist)
    + card('Comisión fuera del watch-list', s.non_watchlist);
}

function renderGrid(lakes) {
  const g = $('#grid'); g.innerHTML = '';
  $('#empty').classList.toggle('hidden', lakes.length > 0);
  for (const x of lakes) {
    const m = x.mine || {};
    const el = document.createElement('div');
    el.className = 'bg-white rounded-2xl border shadow-card overflow-hidden fade-in '
      + (x.wl ? 'border-amber-300' : 'border-slate-200')
      + (m.is_real != null ? ' ring-2 ring-emerald-400' : '');
    el.innerHTML = `
      <div class="relative aspect-square bg-slate-900">
        <img loading="lazy" src="/thumbs/${x.thumb}" class="w-full h-full object-cover" alt="">
        <div class="ring"></div><div class="crosshair"></div>
        <div class="absolute top-2 left-2 flex gap-1">
          ${x.wl ? '<span class="text-[10px] font-bold px-1.5 py-0.5 rounded bg-amber-400 text-amber-950">WATCH</span>' : ''}
          ${x.known ? '<span class="text-[10px] font-bold px-1.5 py-0.5 rounded bg-red-500 text-white">GLOF</span>' : ''}
          ${x.off ? '<span class="text-[10px] font-bold px-1.5 py-0.5 rounded bg-slate-900/70 text-white">off</span>' : ''}
        </div>
      </div>
      <div class="p-3">
        <div class="flex items-center justify-between">
          <span class="font-semibold text-sm capitalize">${x.area.replace(/_/g, ' ')}</span>
          <span class="text-xs text-ink-500 tabular-nums">${(x.score ?? 0).toFixed(2)}</span>
        </div>
        <div class="text-[11px] text-ink-500 mt-0.5 tabular-nums">
          ${(x.area_ha ?? 0).toFixed(1)} ha · ${Math.round((x.dist ?? 0) / 1000)} km al hielo · ${Math.round(x.elev ?? 0)} m
        </div>
        <a class="text-[11px] text-brand-600 hover:underline" target="_blank"
           href="https://earth.google.com/web/@${x.lat},${x.lon},1000a,2500d,35y,0h,0t,0r">Google Earth</a>
        <div class="mt-2.5 grid grid-cols-2 gap-1.5">
          <button data-act="Y" class="yn py-1.5 rounded-lg text-sm font-medium border ${m.is_real === true ? 'bg-emerald-600 text-white border-emerald-600' : 'border-slate-300 hover:bg-slate-50'}">Lago</button>
          <button data-act="N" class="yn py-1.5 rounded-lg text-sm font-medium border ${m.is_real === false ? 'bg-red-600 text-white border-red-600' : 'border-slate-300 hover:bg-slate-50'}">No es</button>
        </div>
        <select class="ty mt-1.5 w-full text-sm rounded-lg border border-slate-300 px-2 py-1.5 bg-white">
          ${TYPES.map(t => `<option value="${t}" ${m.feature_type === t ? 'selected' : ''}>${t || '— tipo —'}</option>`).join('')}
        </select>
        <select class="ob mt-1.5 w-full text-sm rounded-lg border border-slate-300 px-2 py-1.5 bg-white"
                title="Evidencia geomórfica de un vaciamiento pasado: morrena rota, canal de brecha, abanico de descarga">
          <option value="" ${m.outburst_evidence == null ? 'selected' : ''}>— evidencia de GLOF pasado —</option>
          <option value="Y" ${m.outburst_evidence === true ? 'selected' : ''}>Sí: morrena rota / brecha</option>
          <option value="N" ${m.outburst_evidence === false ? 'selected' : ''}>No se observa</option>
        </select>
        <input class="nt mt-1.5 w-full text-sm rounded-lg border border-slate-300 px-2 py-1.5"
               placeholder="nota (opcional)" value="${(m.note || '').replace(/"/g, '&quot;')}">
      </div>`;
    const save = patch => api('/label', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(Object.assign({
        reviewer, lake_key: x.lake_key, is_real: m.is_real,
        feature_type: m.feature_type, confidence: m.confidence,
        outburst_evidence: m.outburst_evidence ?? null, note: m.note
      }, patch))
    });
    el.querySelectorAll('.yn').forEach(b => b.onclick = async () => {
      m.is_real = b.dataset.act === 'Y'; x.mine = m;
      await save({ is_real: m.is_real });
      load();
    });
    el.querySelector('.ty').onchange = e => { m.feature_type = e.target.value; x.mine = m; save({ feature_type: m.feature_type }); };
    el.querySelector('.ob').onchange = e => {
      const v = e.target.value;
      m.outburst_evidence = v === 'Y' ? true : v === 'N' ? false : null; x.mine = m;
      save({ outburst_evidence: m.outburst_evidence });
    };
    el.querySelector('.nt').oninput = e => { m.note = e.target.value; x.mine = m; };
    el.querySelector('.nt').onblur = () => save({ note: m.note });
    g.appendChild(el);
  }
}

// filters
document.querySelectorAll('.filt').forEach(b => b.onclick = () => {
  filter = b.dataset.f;
  document.querySelectorAll('.filt').forEach(x => x.className = 'filt px-3 py-1.5 rounded-md font-medium');
  b.className = 'filt px-3 py-1.5 rounded-md font-medium bg-brand-500 text-white';
  load();
});

$('#loginBtn').onclick = () => { $('#nameIn').value = reviewer; showModal(true); };
$('#enter').onclick = () => { const v = $('#nameIn').value.trim(); if (v) { setReviewer(v); showModal(false); load(); } };
$('#nameIn').addEventListener('keydown', e => { if (e.key === 'Enter') $('#enter').click(); });

setReviewer(reviewer);
document.querySelector('.filt[data-f="watchlist"]').className = 'filt px-3 py-1.5 rounded-md font-medium bg-brand-500 text-white';
load();
