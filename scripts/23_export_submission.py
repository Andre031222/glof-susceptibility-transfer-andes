import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / 'paper' / 'figures'
MAIN = [('Fig1', 'Fig01_study_area'), ('Fig2', 'Fig09_event_evidence'), ('Fig3', 'Fig02_inventory'),
        ('Fig4', 'Fig04_model_performance'), ('Fig5', 'Fig05_distance_learning'), ('Fig6', 'Fig06_transfer_ranks'),
        ('Fig7', 'Fig07_aoa_regions'), ('Fig8', 'Fig08_shap')]
APPENDIX = ['Fig03_nn_distances', 'FigA1_correlogram', 'FigA2_sensitivity']
TARGETS = {
    'nh': (PROJECT_ROOT / 'paper' / 'FINAL_NH' / 'figures',
           MAIN + [(f'Fig{i}', n) for i, n in enumerate(APPENDIX, start=9)], 600),
    'cea': (PROJECT_ROOT / 'paper' / 'alt_cea_elsevier' / 'FINAL' / 'figures',
            MAIN + [(f'FigA{i}', n) for i, n in enumerate(APPENDIX, start=1)], 500),
}
DPI = 600
PHOTOS = {'Fig01_study_area', 'Fig09_event_evidence'}


def export(target):
    out_dir, order, photo_dpi = TARGETS[target]
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None
    for label, name in order:
        src = FIG_DIR / f'{name}.png'
        if not src.exists():
            sys.exit(f'[ERROR] {src} not found; run scripts/20_figures.R, 22_maps.R and 24_event_evidence.py first')
        img = Image.open(src).convert('RGB')
        if name in PHOTOS:
            img = img.resize((round(img.width * photo_dpi / DPI), round(img.height * photo_dpi / DPI)),
                             Image.LANCZOS)
            out = out_dir / f'{label}.jpg'
            img.save(out, quality=95, dpi=(photo_dpi, photo_dpi))
            if target == 'nh':
                img.save(FIG_DIR / f'{name}.jpg', quality=95, dpi=(photo_dpi, photo_dpi))
            stale = out.with_suffix('.tif')
            if stale.exists():
                stale.unlink()
            dpi = photo_dpi
        else:
            out = out_dir / f'{label}.tif'
            img.save(out, compression='tiff_lzw', dpi=(DPI, DPI))
            dpi = DPI
        print(f'  {name} -> {out.relative_to(PROJECT_ROOT)}  {img.width / dpi * 25.4:.0f} x '
              f'{img.height / dpi * 25.4:.0f} mm  {out.stat().st_size / 1e6:.1f} MB')


def main():
    targets = sys.argv[1:] or ['nh']
    for t in targets:
        if t not in TARGETS:
            sys.exit(f'unknown target {t}; use {", ".join(TARGETS)}')
        export(t)


if __name__ == '__main__':
    main()
