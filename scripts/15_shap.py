import os
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get('GLOF_ROOT') or Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings('ignore')

MODELS_DIR = PROJECT_ROOT / 'models'
LAKE_TABLE = PROJECT_ROOT / 'data' / 'processed' / 'labeled' / 'lake_table.csv'
MODEL_FILE = MODELS_DIR / 'primary_model.joblib'
OOF_FILE = MODELS_DIR / 'oof_scores.csv'
RANDOM_STATE = 42


def shap_values(model, X_imp, n_sample):
    import shap
    rng = np.random.RandomState(RANDOM_STATE)
    idx = (rng.choice(X_imp.shape[0], n_sample, replace=False)
           if X_imp.shape[0] > n_sample else np.arange(X_imp.shape[0]))
    Xs = X_imp[idx]
    try:
        return Xs, shap.TreeExplainer(model).shap_values(Xs), 'tree'
    except Exception:
        Xs = Xs[:min(len(Xs), 500)]
        background = shap.sample(X_imp, 100, random_state=RANDOM_STATE)
        explainer = shap.Explainer(lambda z: model.predict_proba(z)[:, 1], background)
        return Xs, explainer(Xs).values, 'permutation'


def export_shap_long(model, X_imp, feature_cols, n_sample=2000, top_k=15):
    Xs, sv, kind = shap_values(model, X_imp, n_sample)
    print(f'SHAP explainer: {kind}')
    if isinstance(sv, list):
        sv = sv[1]
    elif getattr(sv, 'ndim', 2) == 3:
        sv = sv[:, :, 1]
    mean_abs = np.abs(sv).mean(0)
    order = np.argsort(-mean_abs)[:top_k]
    frames = []
    for j in order:
        v = Xs[:, j]
        lo, hi = np.nanpercentile(v, [1, 99])
        scaled = np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)
        frames.append(pd.DataFrame({
            'feature': feature_cols[j],
            'shap': np.round(sv[:, j], 5),
            'value_scaled': np.round(scaled, 4),
            'mean_abs': mean_abs[j],
        }))
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(MODELS_DIR / 'shap_values_long.csv', index=False)
    imp = pd.DataFrame({'feature': [feature_cols[j] for j in order],
                        'importance': mean_abs[order]})
    imp.to_csv(MODELS_DIR / 'shap_importance.csv', index=False)
    print(f'exported shap_values_long.csv ({len(out)} rows, top {top_k}) and shap_importance.csv')


def main():
    for f in (LAKE_TABLE, MODEL_FILE, OOF_FILE):
        if not f.exists():
            sys.exit(f'[ERROR] {f} not found; run 08_lake_table.py and 10_train_validate.py first')
    bundle = joblib.load(MODEL_FILE)
    feature_cols = list(bundle['feature_cols'])
    lakes = pd.read_csv(LAKE_TABLE, low_memory=False)
    keys = pd.read_csv(OOF_FILE, usecols=['lake_key'])['lake_key']
    lakes = lakes[lakes['lake_phys'].isin(keys)].reset_index(drop=True)
    missing = [c for c in feature_cols if c not in lakes.columns]
    if missing:
        raise ValueError(f'lake_table.csv lacks model features {missing}; re-run 10')
    X = lakes[feature_cols].values.astype(float)
    pipe = bundle['model']
    print(f'lakes={len(lakes)} pos={int(lakes["label"].sum())} model={bundle["name"]} '
          f'features={len(feature_cols)}')
    X_imp = pipe.named_steps['imputer'].transform(X)
    export_shap_long(pipe.named_steps['model'], X_imp, feature_cols)
    print('done')


if __name__ == '__main__':
    main()
