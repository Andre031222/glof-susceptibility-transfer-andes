import csv
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = PROJECT_ROOT / 'docs' / 'validation_app'
THUMB_DIR = APP_DIR / 'thumbs'
MONT_DIR = PROJECT_ROOT / 'docs' / 'montages'
SHEET = PROJECT_ROOT / 'docs' / 'inventory_validation_sheet.csv'

COLS, ROWS = 4, 5
CELL = 300
LABEL_H = 26
PER = COLS * ROWS


def font(sz):
    for p in ['/usr/share/fonts/TTF/DejaVuSans-Bold.ttf',
              '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf']:
        if Path(p).exists():
            return ImageFont.truetype(p, sz)
    return ImageFont.load_default()


def num(v, d=0.0):
    try:
        return float(v)
    except (ValueError, TypeError):
        return d


def main():
    MONT_DIR.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(SHEET)))
    for i, r in enumerate(rows):
        r['idx'] = i
    rows = [r for r in rows if num(r['lat']) and num(r['lon'])]
    rows.sort(key=lambda r: (-int(r['in_watchlist_top2pct']),
                             r['area_name'], -num(r['model_score'])))

    f_idx, f_tag = font(20), font(15)
    order = []
    montage = []
    page = 0
    for n, r in enumerate(rows):
        if n % PER == 0:
            if montage:
                save(montage, page, f_idx, f_tag)
                page += 1
            montage = []
        montage.append(r)
        order.append(r['idx'])
    if montage:
        save(montage, page, f_idx, f_tag)

    json.dump(order, open(MONT_DIR / 'order.json', 'w'))
    print(f'wrote {page+1} montages to {MONT_DIR}')
    print(f'total lakes: {len(rows)}  ({sum(int(r["in_watchlist_top2pct"]) for r in rows)} watch-list first)')


def save(cells, page, f_idx, f_tag):
    W = COLS * CELL
    H = ROWS * (CELL + LABEL_H)
    canvas = Image.new('RGB', (W, H), (15, 20, 25))
    d = ImageDraw.Draw(canvas)
    for k, r in enumerate(cells):
        c, rr = k % COLS, k // COLS
        x0, y0 = c * CELL, rr * (CELL + LABEL_H)
        tp = THUMB_DIR / f'{r["idx"]}.jpg'
        if tp.exists():
            im = Image.open(tp).convert('RGB').resize((CELL, CELL))
            canvas.paste(im, (x0, y0 + LABEL_H))
        cx, cy = x0 + CELL // 2, y0 + LABEL_H + CELL // 2
        d.ellipse([cx - 16, cy - 16, cx + 16, cy + 16], outline=(255, 40, 40), width=2)
        d.line([cx, cy - 22, cx, cy - 6], fill=(255, 40, 40), width=2)
        d.line([cx, cy + 6, cx, cy + 22], fill=(255, 40, 40), width=2)
        d.line([cx - 22, cy, cx - 6, cy], fill=(255, 40, 40), width=2)
        d.line([cx + 6, cy, cx + 22, cy], fill=(255, 40, 40), width=2)
        wl = int(r['in_watchlist_top2pct'])
        d.rectangle([x0, y0, x0 + CELL, y0 + LABEL_H], fill=(224, 160, 0) if wl else (26, 34, 48))
        tag = (f"#{r['idx']}  {r['area_name'][:10]}  "
               f"{num(r['area_m2'])/1e4:.1f}ha  {num(r['dist_glacier_m'])/1000:.1f}km")
        d.text((x0 + 4, y0 + 4), tag, fill=(255, 255, 255) if not wl else (0, 0, 0), font=f_tag)
    canvas.save(MONT_DIR / f'montage_{page:03d}.png')


if __name__ == '__main__':
    main()
