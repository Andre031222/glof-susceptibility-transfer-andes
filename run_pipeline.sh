#!/bin/bash
set -e

AREA=${1:-all}
PROJECT=$(cd "$(dirname "$0")" && pwd)
VENV=${VENV:-$PROJECT/.venv}

PY="$VENV/bin/python"
if [ ! -x "$PY" ]; then
    echo "No Python at $PY"
    echo "Create it:  python -m venv $PROJECT/.venv && $PROJECT/.venv/bin/pip install -r requirements.txt"
    echo "or point to an existing one:  VENV=/path/to/venv ./run_pipeline.sh"
    exit 1
fi
cd "$PROJECT"

if [ "$AREA" = "all" ]; then
    AREA_ARG="--all"
else
    AREA_ARG="$AREA"
fi

echo "=========================================="
echo " GLOF susceptibility pipeline   area=$AREA"
echo "=========================================="

echo ""
echo "=== 1. Event catalogue and glacier outlines ==="
$PY scripts/00_glof_catalogue.py
$PY scripts/00_rgi_glaciers.py
$PY scripts/00_osm_context.py

echo ""
echo "=== 2. Acquisition (multi-scene mosaics) ==="
$PY scripts/01_download_landsat.py $AREA_ARG
$PY scripts/02_download_s2.py $AREA_ARG

echo ""
echo "=== 3. DEM and terrain derivatives ==="
$PY scripts/03_process_dem.py $AREA_ARG

echo ""
echo "=== 4. Lake detection (identical criteria, both sensors) ==="
$PY scripts/04_detect_lakes.py $AREA_ARG --force

echo ""
echo "=== 5. Features, physical lakes, labels and lake table ==="
$PY scripts/05_lake_features.py
$PY scripts/06_physical_lakes.py
$PY scripts/07_label_events.py
$PY scripts/08_lake_table.py

echo ""
echo "=== 6. Validation folds (kNNDM, R) ==="
Rscript scripts/09_cv_folds.R

echo ""
echo "=== 7. Models, validation and statistics ==="
$PY scripts/10_train_validate.py
$PY scripts/11_spatial_diagnostics.py
$PY scripts/12_paper_stats.py
$PY scripts/13_pu_learning.py
$PY scripts/14_inventory_screen.py
$PY scripts/15_shap.py
$PY scripts/16_coverage_gaps.py
$PY scripts/17_learning_curve.py
$PY scripts/19_cutoff_seeds.py
$PY scripts/validation/v5_inventory_compare.py --reference wood2021
$PY scripts/validation/v5_inventory_compare.py --reference macmanaway2025
$PY scripts/validation/v5_inventory_compare.py --reference zhang2024
$PY scripts/12_paper_stats.py

echo ""
echo "=== 8. Figures and tables (R) ==="
$PY scripts/18_basemap_rgb.py
Rscript scripts/20_figures.R
Rscript scripts/21_tables.R
Rscript scripts/22_maps.R
$PY scripts/23_export_submission.py

echo ""
echo "=========================================="
echo " DONE"
echo "   models/            folds, CV runs, OOF scores and diagnostics"
echo "   paper/figures/     ggplot2 figures"
echo "   paper/tables/      LaTeX table bodies"
echo "   paper/submission/  Fig1-Fig10 TIFF at 600 dpi"
echo "=========================================="
