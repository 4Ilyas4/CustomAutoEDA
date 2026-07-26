"""Universal, lightweight EDA toolkit.

Works on ANY tabular CSV (Titanic, Iris, air-quality data, or anything else),
not just Titanic-shaped data. The old version hard-coded Titanic column names
and used loose substring matching (e.g. "age" would match inside the word
"average" or "percentage"), which made it fragile on other datasets. This
version:

  * Infers column roles generically (numeric / categorical / datetime /
    identifier / constant / free-text) from the data itself, using dtype and
    cardinality — not column names.
  * Infers whether the target is a classification or regression problem
    generically, instead of assuming a Titanic-style binary target.
  * Picks the right statistical test / association measure for whatever
    feature-type x target-type combination shows up (numeric-numeric,
    numeric-categorical, categorical-categorical).
  * Keeps the old Titanic-specific feature engineering (Title from Name,
    Deck from Cabin, FamilySize, Age imputation) but treats it as *optional,
    best-effort enrichment* that only activates on an exact (not substring)
    match against a small alias list, and never raises if it doesn't apply.
  * Auto-excludes identifier-like and constant columns from statistics and
    modeling instead of relying on a column-name guess like "ends with id".
"""
from typing import Optional, Dict, Any, List
import os
import re
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import chi2_contingency, pearsonr, spearmanr, f_oneway
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from sklearn.model_selection import cross_val_predict
from sklearn.metrics import (roc_auc_score, confusion_matrix, classification_report,
                             accuracy_score, roc_curve, auc)
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
try:
    from category_encoders import TargetEncoder
    CAT_ENCODERS_AVAILABLE = True
except Exception:
    CAT_ENCODERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Basic IO
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


# ---------------------------------------------------------------------------
# Optional domain enrichment (Titanic-style), safe on any dataset
# ---------------------------------------------------------------------------

# Exact-match aliases (normalized: lowercased, non-alphanumeric collapsed to
# single spaces). Using exact matches instead of "key in column_name"
# substring checks avoids false positives like "Percentage" matching "age" or
# "Rating" matching "ticket"... wait no "ticket" itself, but e.g. "Category"
# would no longer false-match "cabin"-style checks either.
_DOMAIN_ALIASES: Dict[str, List[str]] = {
    'name': ['name', 'full name', 'passenger name', 'person name'],
    'cabin': ['cabin', 'cabin number'],
    'age': ['age', 'age years'],
    'sibsp': ['sibsp', 'siblings spouses', 'siblings spouses aboard'],
    'parch': ['parch', 'parents children', 'parents children aboard'],
    'ticket': ['ticket', 'ticket number'],
    'fare': ['fare', 'ticket fare'],
    'embarked': ['embarked', 'embarkation port', 'port of embarkation'],
    'sex': ['sex', 'gender'],
}


def _normalize_colname(c: str) -> str:
    c = str(c).strip().lower()
    c = re.sub(r'[^a-z0-9]+', ' ', c)
    return re.sub(r'\s+', ' ', c).strip()


def detect_column_names(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Best-effort detection of a few optional, well-known column roles.

    Uses exact matching on normalized column names against a small alias
    list, so it will not accidentally trigger on unrelated columns in a
    generic dataset (e.g. an air-quality dataset's "AQI Category" column
    will not be mistaken for "cabin" or "category"-adjacent fields).
    Returns a dict of role -> column name or None; every value is commonly
    None for non-Titanic-shaped data, which is expected and fine.
    """
    normalized = {_normalize_colname(c): c for c in df.columns}
    mapping: Dict[str, Optional[str]] = {k: None for k in _DOMAIN_ALIASES}
    for role, aliases in _DOMAIN_ALIASES.items():
        for alias in aliases:
            if alias in normalized:
                mapping[role] = normalized[alias]
                break
    return mapping


def extract_title(df: pd.DataFrame, name_col: str) -> pd.Series:
    s = df.get(name_col)
    if s is None:
        return pd.Series(index=df.index, dtype='object')
    titles = s.fillna('').astype(str).str.extract(r',\s*([^\.]+)\.', expand=False)
    titles = titles.str.strip().replace('', np.nan).fillna('Unknown')
    top = titles.value_counts().nlargest(10).index
    return titles.where(titles.isin(top), 'Other')


def extract_deck(df: pd.DataFrame, cabin_col: str) -> pd.Series:
    s = df.get(cabin_col)
    if s is None:
        return pd.Series(index=df.index, dtype='object')
    decks = s.fillna('Missing').astype(str).str[0]
    return decks.replace('M', 'Missing')


def add_family_size(df: pd.DataFrame, sibsp: str, parch: str) -> pd.Series:
    return df.get(sibsp, 0).fillna(0).astype(int) + df.get(parch, 0).fillna(0).astype(int) + 1


def impute_age(df: pd.DataFrame, age_col: str, strategy: str = 'median') -> pd.Series:
    s = df.get(age_col)
    if s is None:
        return pd.Series(index=df.index, dtype=float)
    fill = s.median() if strategy == 'median' else s.mean()
    return s.fillna(fill)


def apply_domain_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Title / Deck / FamilySize / Age_imputed columns when (and only
    when) the relevant source columns are confidently detected. Never
    raises; a dataset with none of these columns comes back unchanged."""
    df2 = df.copy()
    colmap = detect_column_names(df2)
    try:
        if colmap.get('name'):
            df2['Title'] = extract_title(df2, colmap['name'])
        if colmap.get('cabin'):
            df2['Deck'] = extract_deck(df2, colmap['cabin'])
        sib, parch = colmap.get('sibsp'), colmap.get('parch')
        if sib and parch:
            df2['FamilySize'] = add_family_size(df2, sib, parch)
        elif sib:
            df2['FamilySize'] = df2[sib].fillna(0).astype(int) + 1
        elif parch:
            df2['FamilySize'] = df2[parch].fillna(0).astype(int) + 1
        if colmap.get('age'):
            df2['Age_imputed'] = impute_age(df2, colmap['age'])
            df2['Age_missing_flag'] = df2[colmap['age']].isna().astype(int)
    except Exception:
        # Enrichment is a bonus, never let it break generic analysis.
        return df.copy()
    return df2


# ---------------------------------------------------------------------------
# Generic column-role and target-type inference
# ---------------------------------------------------------------------------

def infer_target_type(y: pd.Series) -> str:
    """Returns 'classification' or 'regression', inferred from dtype and
    cardinality rather than assumed."""
    y_nonnull = y.dropna()
    if len(y_nonnull) == 0:
        return 'classification'
    if pd.api.types.is_bool_dtype(y_nonnull):
        return 'classification'
    if pd.api.types.is_numeric_dtype(y_nonnull):
        n_unique = y_nonnull.nunique()
        looks_discrete = n_unique <= 20 and np.all(np.mod(y_nonnull, 1) == 0)
        return 'classification' if looks_discrete else 'regression'
    return 'classification'


def _looks_like_datetime(s: pd.Series, min_success_ratio: float = 0.9) -> bool:
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False
    sample = s.dropna()
    if len(sample) == 0:
        return False
    if len(sample) > 200:
        sample = sample.sample(200, random_state=0)
    try:
        parsed = pd.to_datetime(sample, errors='coerce')
    except Exception:
        return False
    return parsed.notna().mean() >= min_success_ratio


def classify_columns(df: pd.DataFrame, target: str,
                      id_uniqueness_ratio: float = 0.95,
                      max_categorical_unique: int = 50,
                      detect_dates: bool = True) -> Dict[str, str]:
    """Classifies every non-target column into one role:
    'numeric', 'categorical', 'datetime', 'identifier', 'constant', 'text'.

    This replaces the old hard-coded "numeric and nunique>10 => numerical
    else categorical" rule, which had no notion of IDs, dates, or free text
    and would try to one-hot-encode or chi2-test a unique-per-row Name
    column against the target.
    """
    n = len(df)
    roles: Dict[str, str] = {}
    for c in df.columns:
        if c == target:
            continue
        s = df[c]
        nunique = s.nunique(dropna=True)
        if nunique <= 1:
            roles[c] = 'constant'
            continue
        uniqueness_ratio = nunique / n if n else 0
        if pd.api.types.is_datetime64_any_dtype(s):
            roles[c] = 'datetime'
        elif detect_dates and _looks_like_datetime(s):
            roles[c] = 'datetime'
        elif pd.api.types.is_bool_dtype(s):
            roles[c] = 'categorical'
        elif pd.api.types.is_numeric_dtype(s):
            if uniqueness_ratio > id_uniqueness_ratio and nunique > max_categorical_unique:
                roles[c] = 'identifier'
            else:
                roles[c] = 'numeric'
        else:
            if uniqueness_ratio > id_uniqueness_ratio:
                roles[c] = 'identifier'
            elif nunique > max_categorical_unique:
                roles[c] = 'text'
            else:
                roles[c] = 'categorical'
    return roles


def expand_datetime_features(df: pd.DataFrame, col: str) -> pd.DataFrame:
    s = pd.to_datetime(df[col], errors='coerce')
    return pd.DataFrame({
        f'{col}_year': s.dt.year,
        f'{col}_month': s.dt.month,
        f'{col}_day': s.dt.day,
        f'{col}_dayofweek': s.dt.dayofweek,
    }, index=df.index)


def _safe_filename(name: str) -> str:
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(name))


def cramers_v(confusion_matrix: pd.DataFrame) -> Optional[float]:
    if not SCIPY_AVAILABLE:
        return None
    try:
        chi2 = chi2_contingency(confusion_matrix)[0]
        n = confusion_matrix.to_numpy().sum()
        if n == 0:
            return None
        r, k = confusion_matrix.shape
        denom = min(k - 1, r - 1)
        if denom <= 0:
            return None
        return float(np.sqrt((chi2 / n) / denom))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Per-feature analysis (dispatches on feature type x target type)
# ---------------------------------------------------------------------------

def analyze_numerical(df: pd.DataFrame, feature: str, target: str,
                       target_type: str = 'auto', outdir: Optional[str] = None) -> Dict[str, Any]:
    s = df[feature]
    t = df[target]
    res: Dict[str, Any] = {'feature': feature, 'dtype': str(s.dtype),
                            'n_missing': int(s.isna().sum())}
    if s.dropna().shape[0] > 0:
        res['min'] = float(s.min())
        res['max'] = float(s.max())
        res['mean'] = float(s.mean())
    else:
        res['min'] = res['max'] = res['mean'] = None

    if target_type == 'auto':
        target_type = infer_target_type(t)
    res['target_type_used'] = target_type

    valid = pd.concat([s, t], axis=1).dropna()

    if target_type == 'regression' and pd.api.types.is_numeric_dtype(t):
        if SCIPY_AVAILABLE and valid.shape[0] > 1:
            try:
                res['pearson'] = float(pearsonr(valid[feature], valid[target])[0])
                res['spearman'] = float(spearmanr(valid[feature], valid[target])[0])
            except Exception:
                res['pearson'] = res['spearman'] = None
        else:
            res['pearson'] = res['spearman'] = None
        if SKLEARN_AVAILABLE and valid.shape[0] > 1:
            try:
                mi = mutual_info_regression(valid[[feature]].values, valid[target].values, random_state=0)[0]
                res['mutual_info_regression'] = float(mi)
            except Exception:
                res['mutual_info_regression'] = None
    else:
        # Numeric feature vs categorical/classification target: ANOVA + MI.
        try:
            groups = [g[feature].values for _, g in valid.groupby(target) if len(g) > 0]
            if SCIPY_AVAILABLE and len(groups) > 1:
                _, p = f_oneway(*groups)
                res['anova_p'] = float(p)
            else:
                res['anova_p'] = None
        except Exception:
            res['anova_p'] = None
        if SKLEARN_AVAILABLE and valid.shape[0] > 1:
            try:
                targ_enc = LabelEncoder().fit_transform(valid[target].astype(str))
                mi = mutual_info_classif(valid[[feature]].values, targ_enc, random_state=0)[0]
                res['mutual_info_classif'] = float(mi)
            except Exception:
                res['mutual_info_classif'] = None

    if outdir:
        try:
            os.makedirs(outdir, exist_ok=True)
            plt.figure(figsize=(6, 4))
            sns.histplot(s.dropna(), kde=True)
            plt.title(f'Distribution: {feature}')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'{_safe_filename(feature)}_hist.png'))
            plt.close()
        except Exception:
            plt.close('all')

    return res


def analyze_categorical(df: pd.DataFrame, feature: str, target: str,
                         target_type: str = 'auto', outdir: Optional[str] = None) -> Dict[str, Any]:
    s = df[feature]
    t = df[target]
    res: Dict[str, Any] = {'feature': feature, 'dtype': str(s.dtype),
                            'n_unique': int(s.nunique(dropna=False)),
                            'n_missing': int(s.isna().sum())}
    vc = s.value_counts(dropna=False)
    res['top_values'] = {str(k): int(v) for k, v in vc.head(20).items()}

    if target_type == 'auto':
        target_type = infer_target_type(t)
    res['target_type_used'] = target_type

    valid = pd.concat([s, t], axis=1).dropna()

    if target_type == 'classification':
        if SCIPY_AVAILABLE and s.nunique(dropna=True) >= 2 and valid.shape[0] > 0:
            try:
                ct = pd.crosstab(valid[feature], valid[target])
                chi2, p, _, _ = chi2_contingency(ct)
                res['chi2_p'] = float(p)
                res['cramers_v'] = cramers_v(ct)
            except Exception:
                res['chi2_p'] = None
                res['cramers_v'] = None
        if SKLEARN_AVAILABLE and valid.shape[0] > 1:
            try:
                feat_enc = LabelEncoder().fit_transform(valid[feature].astype(str))
                targ_enc = LabelEncoder().fit_transform(valid[target].astype(str))
                mi = mutual_info_classif(feat_enc.reshape(-1, 1), targ_enc, random_state=0)[0]
                res['mutual_info_classif'] = float(mi)
            except Exception:
                res['mutual_info_classif'] = None
    else:
        # Categorical feature vs regression (numeric) target.
        try:
            groups = [g[target].values for _, g in valid.groupby(feature) if len(g) > 0]
            if SCIPY_AVAILABLE and len(groups) > 1:
                _, p = f_oneway(*groups)
                res['anova_p'] = float(p)
            else:
                res['anova_p'] = None
        except Exception:
            res['anova_p'] = None
        if SKLEARN_AVAILABLE and valid.shape[0] > 1:
            try:
                feat_enc = LabelEncoder().fit_transform(valid[feature].astype(str))
                mi = mutual_info_regression(feat_enc.reshape(-1, 1), valid[target].values, random_state=0)[0]
                res['mutual_info_regression'] = float(mi)
            except Exception:
                res['mutual_info_regression'] = None

    if outdir:
        try:
            os.makedirs(outdir, exist_ok=True)
            plt.figure(figsize=(8, 4))
            vc.head(40).plot(kind='bar')
            plt.title(f'Distribution of {feature}')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, f'{_safe_filename(feature)}_dist.png'))
            plt.close()
        except Exception:
            plt.close('all')

    return res


def summarize_skipped_column(df: pd.DataFrame, feature: str, role: str) -> Dict[str, Any]:
    """Lightweight summary for identifier / constant / text columns that are
    excluded from statistical testing and modeling (too high-cardinality or
    no variance to be meaningful), so they're still visible in the report."""
    s = df[feature]
    res: Dict[str, Any] = {
        'feature': feature, 'role': role, 'dtype': str(s.dtype),
        'n_unique': int(s.nunique(dropna=False)), 'n_missing': int(s.isna().sum()),
    }
    if role == 'text':
        vc = s.value_counts(dropna=False)
        res['top_values'] = {str(k): int(v) for k, v in vc.head(10).items()}
    return res


# ---------------------------------------------------------------------------
# Feature encoding for modeling
# ---------------------------------------------------------------------------

def encode_features_for_model(df: pd.DataFrame, categorical_threshold: int = 10) -> pd.DataFrame:
    """Encodes a dataframe into a numeric matrix suitable for ML.

    - Datetime columns -> expanded into year/month/day/dayofweek numerics.
    - Low-cardinality categoricals (<= threshold) -> one-hot via get_dummies.
    - High-cardinality categoricals -> frequency encoding.
    - Numeric columns kept as-is (NaNs filled by caller).
    """
    df2 = df.copy()

    datetime_cols = [c for c in df2.columns if pd.api.types.is_datetime64_any_dtype(df2[c])
                      or (not pd.api.types.is_numeric_dtype(df2[c]) and _looks_like_datetime(df2[c]))]
    for c in datetime_cols:
        expanded = expand_datetime_features(df2, c)
        df2 = pd.concat([df2.drop(columns=[c]), expanded], axis=1)

    cats = [c for c in df2.columns if not pd.api.types.is_numeric_dtype(df2[c])]
    to_concat = []
    keep = []

    for c in cats:
        nunique = df2[c].nunique(dropna=True)
        if nunique <= categorical_threshold:
            d = pd.get_dummies(df2[c].astype(str), prefix=c, dummy_na=False)
            to_concat.append(d)
        else:
            filled = df2[c].fillna('___NA___').astype(str)
            counts = filled.value_counts()
            df2[c + '_freq'] = filled.map(counts).astype(float)
            keep.append(c + '_freq')

    numeric_cols = [c for c in df2.columns if pd.api.types.is_numeric_dtype(df2[c])]
    keep.extend(numeric_cols)
    res = df2[keep].copy()
    if to_concat:
        res = pd.concat([res] + to_concat, axis=1)
    return res


# ---------------------------------------------------------------------------
# Baseline model
# ---------------------------------------------------------------------------

def baseline_model(df: pd.DataFrame, target: str, features: Optional[list] = None,
                    outdir: Optional[str] = None, cv: int = 5,
                    encode_categoricals: bool = True, one_hot_thresh: int = 10,
                    drop_columns: Optional[list] = None,
                    column_roles: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Trains a simple RandomForest baseline (classifier or regressor,
    chosen generically from the target's inferred type) and returns CV score
    + feature importances."""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError('scikit-learn is required for baseline_model')

    y = df[target]
    is_classif = infer_target_type(y) == 'classification'

    df = df.copy()
    if drop_columns:
        df = df.drop(columns=[c for c in drop_columns if c in df.columns], errors='ignore')

    if features is None:
        if column_roles is None:
            column_roles = classify_columns(df, target)
        # Exclude identifier / constant columns automatically; they add
        # noise or leak row-order information rather than signal.
        features = [c for c, role in column_roles.items()
                    if role not in ('identifier', 'constant') and c in df.columns]

    df_feats = df[features].copy()

    if encode_categoricals:
        X = encode_features_for_model(df_feats, categorical_threshold=one_hot_thresh)
    else:
        X = df_feats.select_dtypes(include=[np.number]).copy()

    X = X.fillna(-999)
    res: Dict[str, Any] = {'features': list(X.columns), 'target_type': 'classification' if is_classif else 'regression'}

    if X.shape[1] == 0:
        res['error'] = 'No usable features after encoding/exclusion.'
        return res

    mask = y.notna()
    X, y = X[mask], y[mask]

    if is_classif:
        model = RandomForestClassifier(n_estimators=100, random_state=0)
        scoring = 'accuracy'
        min_class_count = y.value_counts().min() if y.nunique() > 0 else 0
        eff_cv = max(2, min(cv, int(min_class_count))) if min_class_count >= 2 else 0
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        scoring = 'r2'
        eff_cv = max(2, min(cv, len(y))) if len(y) >= 2 else 0

    if eff_cv < 2:
        res['error'] = 'Not enough samples/classes to run cross-validation.'
        return res

    try:
        scores = cross_val_score(model, X, y, cv=eff_cv, scoring=scoring)
        res['cv_folds_used'] = eff_cv
        res['cv_mean'] = float(scores.mean())
        res['cv_std'] = float(scores.std())

        model.fit(X, y)
        if hasattr(model, 'feature_importances_'):
            res['importances'] = dict(sorted(
                zip(X.columns.tolist(), model.feature_importances_.tolist()),
                key=lambda kv: kv[1], reverse=True))

        # Cross-validated predictions for additional metrics
        try:
            y_pred = cross_val_predict(model, X, y, cv=eff_cv)
            res['cv_accuracy'] = float(accuracy_score(y, y_pred))
            res['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
            res['classification_report'] = classification_report(y, y_pred, output_dict=True)
        except Exception:
            pass

        # Probabilities -> ROC AUC (if possible)
        try:
            if hasattr(model, 'predict_proba'):
                y_proba = cross_val_predict(model, X, y, cv=eff_cv, method='predict_proba')
                if y_proba is not None:
                    if y_proba.ndim == 1 or y_proba.shape[1] == 2:
                        # binary
                        probs = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                        res['roc_auc'] = float(roc_auc_score(y, probs))
                    else:
                        res['roc_auc_ovr_macro'] = float(roc_auc_score(y, y_proba, multi_class='ovr', average='macro'))
        except Exception:
            pass

        if outdir:
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, 'baseline_summary.json'), 'w', encoding='utf8') as f:
                json.dump(res, f, indent=2)
    except Exception as e:
        res['error'] = str(e)

    return res


def run_advanced_pipeline(df: pd.DataFrame, target: str, outdir: Optional[str] = None,
                          one_hot_thresh: int = 10, cv: int = 5, param_grid: Optional[Dict[str, List[Any]]] = None):
    """Builds a pipeline: OneHot for low-cardinality categoricals + TargetEncoder (or frequency) for high-cardinality,
    then GridSearch over LightGBM (if available) or sklearn's HistGradientBoosting. Saves best model results and metrics.
    """
    if outdir:
        os.makedirs(outdir, exist_ok=True)

    roles = classify_columns(df, target)
    categorical = [c for c, r in roles.items() if r == 'categorical']
    numeric = [c for c, r in roles.items() if r == 'numeric']

    low_card = [c for c in categorical if df[c].nunique(dropna=True) <= one_hot_thresh]
    high_card = [c for c in categorical if df[c].nunique(dropna=True) > one_hot_thresh]

    # Prepare transformers
    transformers = []
    if low_card:
        transformers.append(('ohe', OneHotEncoder(handle_unknown='ignore', sparse=False), low_card))

    # For high-cardinality, use TargetEncoder if available, else frequency encoding (we will apply TargetEncoder inside pipeline)
    preproc_steps = []
    if CAT_ENCODERS_AVAILABLE and high_card:
        # ColumnTransformer applies TargetEncoder to high_card directly
        transformers.append(('te', TargetEncoder(cols=high_card), high_card))

    column_transformer = ColumnTransformer(transformers=transformers, remainder='passthrough')

    # Choose estimator
    try:
        from lightgbm import LGBMClassifier
        estimator = LGBMClassifier(random_state=0)
        default_grid = {'estimator__n_estimators': [50, 100], 'estimator__learning_rate': [0.05, 0.1]}
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier
        estimator = HistGradientBoostingClassifier(random_state=0)
        default_grid = {'estimator__max_iter': [100, 200], 'estimator__learning_rate': [0.05, 0.1]}

    pipe = Pipeline([('pre', column_transformer), ('estimator', estimator)])

    if param_grid is None:
        param_grid = default_grid

    from sklearn.model_selection import GridSearchCV
    y = df[target]
    mask = y.notna()
    X = df.drop(columns=[target])[mask]
    y = y[mask]

    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc' if infer_target_type(y)=='classification' else 'r2', n_jobs=-1)
    gs.fit(X, y)

    best = gs.best_estimator_
    results = {'best_params': gs.best_params_, 'best_score': float(gs.best_score_)}

    # cross-validated predictions for metrics
    try:
        y_pred = cross_val_predict(best, X, y, cv=cv)
        y_proba = None
        if hasattr(best, 'predict_proba'):
            y_proba = cross_val_predict(best, X, y, cv=cv, method='predict_proba')
        results['confusion_matrix'] = confusion_matrix(y, y_pred).tolist()
        results['classification_report'] = classification_report(y, y_pred, output_dict=True)
        if y_proba is not None:
            if y_proba.ndim == 1 or (hasattr(y_proba, 'shape') and y_proba.shape[1]==2):
                probs = y_proba[:,1] if y_proba.ndim>1 else y_proba
                results['roc_auc'] = float(roc_auc_score(y, probs))
            else:
                results['roc_auc_ovr_macro'] = float(roc_auc_score(y, y_proba, multi_class='ovr', average='macro'))
    except Exception as e:
        results['eval_error'] = str(e)

    # Save plots: confusion matrix and ROC curves
    if outdir:
        try:
            os.makedirs(outdir, exist_ok=True)
            # Confusion matrix plot
            cm = confusion_matrix(y, cross_val_predict(best, X, y, cv=cv))
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True')
            plt.xlabel('Pred')
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, 'confusion_matrix.png'))
            plt.close()

            # ROC curves
            if y_proba is not None:
                classes = np.unique(y)
                # binarize true labels
                try:
                    y_bin = label_binarize(y.astype(str), classes=[str(c) for c in classes])
                except Exception:
                    y_bin = None

                plt.figure(figsize=(7, 6))
                if y_proba.ndim == 1 or (hasattr(y_proba, 'shape') and y_proba.shape[1] == 2):
                    # binary
                    probs = y_proba[:, 1] if y_proba.ndim > 1 else y_proba
                    fpr, tpr, _ = roc_curve(y, probs)
                    roc_auc = auc(fpr, tpr)
                    plt.plot(fpr, tpr, label=f'ROC (AUC = {roc_auc:.3f})')
                else:
                    # multiclass: plot per-class
                    for i, cls in enumerate(classes):
                        if y_bin is None:
                            try:
                                y_bin = label_binarize(y, classes=classes)
                            except Exception:
                                y_bin = None
                                break
                        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
                        roc_auc = auc(fpr, tpr)
                        plt.plot(fpr, tpr, label=f'{cls} (AUC={roc_auc:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', linewidth=0.8)
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curves')
                plt.legend(loc='lower right')
                plt.tight_layout()
                plt.savefig(os.path.join(outdir, 'roc_curves.png'))
                plt.close()
        except Exception:
            plt.close('all')

    if outdir:
        with open(os.path.join(outdir, 'pipeline_grid_results.json'), 'w', encoding='utf8') as f:
            json.dump(results, f, indent=2)

    return results


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _association_strength(feat_res: Dict[str, Any]) -> Optional[float]:
    for key in ('cramers_v', 'mutual_info_classif', 'mutual_info_regression'):
        v = feat_res.get(key)
        if v is not None:
            return float(v)
    for key in ('pearson', 'spearman'):
        v = feat_res.get(key)
        if v is not None:
            return abs(float(v))
    return None


def write_markdown_report(results: Dict[str, Any], outdir: str, target: str) -> str:
    lines = [f"# EDA Report", "", f"Target column: **{target}**",
              f"Rows: {results.get('n_rows')} | Columns: {results.get('n_columns')}",
              f"Inferred target type: **{results.get('target_type')}**", ""]

    ranked = []
    for feat, r in results.get('features', {}).items():
        strength = _association_strength(r) if isinstance(r, dict) else None
        if strength is not None:
            ranked.append((feat, strength))
    ranked.sort(key=lambda kv: kv[1], reverse=True)

    if ranked:
        lines.append("## Features most associated with the target")
        lines.append("")
        lines.append("| Feature | Association strength |")
        lines.append("|---|---|")
        for feat, strength in ranked[:20]:
            lines.append(f"| {feat} | {strength:.4f} |")
        lines.append("")

    skipped = results.get('skipped_columns', {})
    if skipped:
        lines.append("## Columns excluded from deep analysis")
        lines.append("")
        lines.append("| Column | Reason |")
        lines.append("|---|---|")
        for feat, info in skipped.items():
            lines.append(f"| {feat} | {info.get('role')} |")
        lines.append("")

    baseline = results.get('baseline')
    if baseline:
        lines.append("## Baseline RandomForest model")
        lines.append("")
        if 'error' in baseline:
            lines.append(f"Could not train a baseline model: {baseline['error']}")
        else:
            metric = 'accuracy' if baseline.get('target_type') == 'classification' else 'R^2'
            lines.append(f"CV {metric}: {baseline.get('cv_mean'):.4f} (+/- {baseline.get('cv_std'):.4f}), "
                          f"folds={baseline.get('cv_folds_used')}")
            lines.append("")
            top_imp = list(baseline.get('importances', {}).items())[:10]
            if top_imp:
                lines.append("Top features by importance:")
                lines.append("")
                for feat, imp in top_imp:
                    lines.append(f"- {feat}: {imp:.4f}")
        lines.append("")

    path = os.path.join(outdir, 'report.md')
    with open(path, 'w', encoding='utf8') as f:
        f.write('\n'.join(lines))
    return path


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def analyze(df: pd.DataFrame, target: str, outdir: Optional[str] = None, max_features: int = 50,
            run_baseline: bool = False, enable_domain_enrichment: bool = True,
            id_uniqueness_ratio: float = 0.95, max_categorical_unique: int = 50) -> Dict[str, Any]:
    """Runs a full EDA pass on ANY dataset: infers column roles and target
    type from the data, runs the appropriate association test per feature,
    optionally adds Titanic-style bonus features when they apply, and
    (optionally) trains a baseline model. Works unchanged for Titanic, Iris,
    air-quality data, or any other tabular CSV."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found. Available columns: {list(df.columns)}")

    df2 = apply_domain_enrichment(df) if enable_domain_enrichment else df.copy()

    target_type = infer_target_type(df2[target])
    roles = classify_columns(df2, target, id_uniqueness_ratio=id_uniqueness_ratio,
                              max_categorical_unique=max_categorical_unique)

    results: Dict[str, Any] = {
        'n_rows': int(df2.shape[0]), 'n_columns': int(df2.shape[1]),
        'target_type': target_type, 'features': {}, 'skipped_columns': {},
    }

    analyzable_cols = [c for c, role in roles.items() if role in ('numeric', 'categorical', 'datetime')]
    skipped_cols = [c for c, role in roles.items() if role in ('identifier', 'constant', 'text')]

    if max_features:
        analyzable_cols = analyzable_cols[:max_features]

    for c in analyzable_cols:
        role = roles[c]
        try:
            if role == 'datetime':
                s = pd.to_datetime(df2[c], errors='coerce')
                results['features'][c] = {
                    'feature': c, 'role': 'datetime', 'n_missing': int(s.isna().sum()),
                    'min_date': str(s.min()) if s.notna().any() else None,
                    'max_date': str(s.max()) if s.notna().any() else None,
                }
                expanded = expand_datetime_features(df2, c)
                for sub_col in expanded.columns:
                    tmp = df2.drop(columns=[]).copy()
                    tmp[sub_col] = expanded[sub_col]
                    if sub_col.endswith('_dayofweek'):
                        results['features'][sub_col] = analyze_categorical(tmp, sub_col, target, target_type, outdir)
                    else:
                        results['features'][sub_col] = analyze_numerical(tmp, sub_col, target, target_type, outdir)
            elif role == 'numeric':
                results['features'][c] = analyze_numerical(df2, c, target, target_type, outdir)
            else:
                results['features'][c] = analyze_categorical(df2, c, target, target_type, outdir)
        except Exception as e:
            results['features'][c] = {'error': str(e)}

    for c in skipped_cols:
        try:
            results['skipped_columns'][c] = summarize_skipped_column(df2, c, roles[c])
        except Exception as e:
            results['skipped_columns'][c] = {'error': str(e)}

    if run_baseline:
        results['baseline'] = baseline_model(df2, target, outdir=outdir, encode_categoricals=True,
                                              one_hot_thresh=10, column_roles=roles)

    if outdir:
        os.makedirs(outdir, exist_ok=True)
        with open(os.path.join(outdir, 'analysis_summary.json'), 'w', encoding='utf8') as f:
            json.dump(results, f, indent=2, default=str)
        write_markdown_report(results, outdir, target)

    return results

