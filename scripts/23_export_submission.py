import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
FIG_DIR = PROJECT_ROOT / 'paper' / 'figures'
OUT_DIR = PROJECT_ROOT / 'paper' / 'submission'
ORDER = ['Fig01_study_area', 'Fig02_inventory', 'Fig03_nn_distances', 'Fig04_model_performance',
         'Fig05_distance_learning', 'Fig06_transfer_ranks', 'Fig07_applicability', 'Fig08_shap',
         'FigA1_correlogram', 'FigA2_sensitivity']
DPI = 600


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.MAX_IMAGE_PIXELS = None
    for i, name in enumerate(ORDER, start=1):
        src = FIG_DIR / f'{name}.png'
        if not src.exists():
            sys.exit(f'[ERROR] {src} not found; run scripts/20_figures.R and 22_maps.R first')
        img = Image.open(src).convert('RGB')
        out = OUT_DIR / f'Fig{i}.tif'
        img.save(out, compression='tiff_lzw', dpi=(DPI, DPI))
        w_mm = img.width / DPI * 25.4
        print(f'  {name} -> {out.name}  {img.width}x{img.height} px  {w_mm:.0f} mm wide at {DPI} dpi')


if __name__ == '__main__':
    main()
