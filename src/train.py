from pathlib import Path
import warnings

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, PrecisionRecallDisplay,
                             RocCurveDisplay, average_precision_score,
                             classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
(ROOT / "models").mkdir(exist_ok=True)
SEED = 42

# ------------------------------------------------------------------ 1. Load
df = pd.read_csv(ROOT / "data" / "healthcare-dataset-stroke-data.csv", na_values=["N/A"])
print("Shape:", df.shape)
print("Missing values:\n", df.isna().sum()[df.isna().sum() > 0])
print("Class balance:\n", df["stroke"].value_counts(normalize=True).round(4))

df = df.drop(columns="id")
df = df[df["gender"] != "Other"].reset_index(drop=True)  # single row

# ------------------------------------------------------------------ 2. EDA
fig, ax = plt.subplots(figsize=(5, 4))
sns.countplot(data=df, x="stroke", ax=ax)
ax.set_title("Class distribution (heavily imbalanced)")
for p in ax.patches:
    ax.annotate(int(p.get_height()), (p.get_x() + p.get_width() / 2, p.get_height()),
                ha="center", va="bottom")
fig.tight_layout(); fig.savefig(FIG / "01_class_balance.png", dpi=150); plt.close(fig)

num_cols = ["age", "avg_glucose_level", "bmi"]
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for a, c in zip(axes, num_cols):
    sns.kdeplot(data=df, x=c, hue="stroke", common_norm=False, fill=True, ax=a)
    a.set_title(f"{c} by stroke")
fig.tight_layout(); fig.savefig(FIG / "02_numeric_distributions.png", dpi=150); plt.close(fig)

cat_cols_eda = ["gender", "hypertension", "heart_disease", "ever_married",
                "work_type", "Residence_type", "smoking_status"]
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for a, c in zip(axes.ravel(), cat_cols_eda):
    (df.groupby(c)["stroke"].mean() * 100).sort_values().plot.barh(ax=a)
    a.set_title(f"Stroke rate (%) by {c}"); a.set_ylabel("")
axes.ravel()[-1].axis("off")
fig.tight_layout(); fig.savefig(FIG / "03_stroke_rate_by_category.png", dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(df.select_dtypes("number").corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
ax.set_title("Correlation matrix")
fig.tight_layout(); fig.savefig(FIG / "04_correlation.png", dpi=150); plt.close(fig)

# ------------------------------------------------------------ 3. Preprocess
X, y = df.drop(columns="stroke"), df["stroke"]
num_features = ["age", "avg_glucose_level", "bmi"]
cat_features = ["gender", "ever_married", "work_type", "Residence_type", "smoking_status"]
bin_features = ["hypertension", "heart_disease"]

pre = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc", StandardScaler())]), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
    ("bin", "passthrough", bin_features),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)
print(f"\nTrain: {X_train.shape}, Test: {X_test.shape}")

# ------------------------------------------------------------ 4. Models
# Imbalance is handled with class weights (no leakage, no synthetic data).
models = {
    "Logistic Regression": lambda: LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                                    class_weight="balanced", random_state=SEED, n_jobs=-1),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=SEED),
}


def make_pipe(name):
    return Pipeline([("pre", pre), ("clf", models[name]())])


def fit(pipe, Xd, yd, name):
    if name == "Gradient Boosting":  # no class_weight param -> use sample weights
        pipe.fit(Xd, yd, clf__sample_weight=compute_sample_weight("balanced", yd))
    else:
        pipe.fit(Xd, yd)
    return pipe


def best_threshold(y_true, proba):
    """Threshold maximising F2 (recall weighted higher - missing a stroke is costlier)."""
    best_t, best_s = 0.5, -1
    for t in np.linspace(0.05, 0.95, 91):
        pr = (proba >= t).astype(int)
        p, r = precision_score(y_true, pr, zero_division=0), recall_score(y_true, pr)
        s = 5 * p * r / max(4 * p + r, 1e-9)
        if s > best_s:
            best_t, best_s = t, s
    return best_t


# Out-of-fold probabilities on TRAIN pick the threshold (test set stays untouched)
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
rows, fitted, thresholds = [], {}, {}
for name in models:
    oof = np.zeros(len(X_train))
    for tr, va in skf.split(X_train, y_train):
        p = fit(make_pipe(name), X_train.iloc[tr], y_train.iloc[tr], name)
        oof[va] = p.predict_proba(X_train.iloc[va])[:, 1]
    thr = best_threshold(y_train.values, oof)
    pipe = fit(make_pipe(name), X_train, y_train, name)
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= thr).astype(int)
    rows.append({"Model": name, "Threshold": round(thr, 2),
                 "ROC-AUC": roc_auc_score(y_test, proba),
                 "PR-AUC": average_precision_score(y_test, proba),
                 "Recall": recall_score(y_test, pred),
                 "Precision": precision_score(y_test, pred, zero_division=0),
                 "F1": f1_score(y_test, pred)})
    fitted[name], thresholds[name] = pipe, thr

results = pd.DataFrame(rows).set_index("Model").round(3)
print("\n=== Test-set results ===\n", results)
results.to_csv(ROOT / "reports" / "model_comparison.csv")

# ------------------------------------------------------------ 5. Best model
best = results["ROC-AUC"].idxmax()
pipe, thr = fitted[best], thresholds[best]
proba = pipe.predict_proba(X_test)[:, 1]
pred = (proba >= thr).astype(int)
print(f"\nBest model (by ROC-AUC): {best}  | threshold = {thr:.2f}")
print(classification_report(y_test, pred, target_names=["No stroke", "Stroke"]))

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
for name, p in fitted.items():
    RocCurveDisplay.from_estimator(p, X_test, y_test, name=name, ax=axes[0])
    PrecisionRecallDisplay.from_estimator(p, X_test, y_test, name=name, ax=axes[1])
axes[0].plot([0, 1], [0, 1], "k--"); axes[0].set_title("ROC curves")
axes[1].set_title("Precision-Recall curves")
ConfusionMatrixDisplay(confusion_matrix(y_test, pred), display_labels=["No stroke", "Stroke"]).plot(
    ax=axes[2], colorbar=False, cmap="Blues")
axes[2].set_title(f"{best} (thr={thr:.2f})")
fig.tight_layout(); fig.savefig(FIG / "05_model_evaluation.png", dpi=150); plt.close(fig)

imp = permutation_importance(pipe, X_test, y_test, scoring="roc_auc", n_repeats=20,
                             random_state=SEED, n_jobs=-1)
imp_s = pd.Series(imp.importances_mean, index=X_test.columns).sort_values()
fig, ax = plt.subplots(figsize=(7, 5))
imp_s.plot.barh(ax=ax); ax.set_title(f"Permutation importance (drop in ROC-AUC)\n{best}")
fig.tight_layout(); fig.savefig(FIG / "06_feature_importance.png", dpi=150); plt.close(fig)
print("\nTop features:\n", imp_s.sort_values(ascending=False).head(5).round(4))

joblib.dump({"pipeline": pipe, "threshold": float(thr)}, ROOT / "models" / "stroke_model.joblib")
print("\nSaved model -> models/stroke_model.joblib")
