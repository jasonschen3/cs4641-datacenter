"""
ML Pipeline: Loading data -> Feature Engineering -> Training -> Evaluation -> Visualization
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
import urllib.request
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, precision_recall_curve, auc,
    classification_report, log_loss, roc_auc_score
)
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({"figure.dpi": 120, "font.size": 11})

FEATURE_COLS = [
    "Population", "pop_density", "median_income",
    "elec_rate", "water_avail", "tax_exempt", "avg_temp",
    "dc_lag_1", "dc_lag_3", "cumulative_dc",
]

FEATURE_GROUPS = {
    "Demographic/Economic": ["Population", "pop_density", "median_income"],
    "Infrastructure":       ["elec_rate", "water_avail"],
    "Policy":               ["tax_exempt"],
    "Environmental":        ["avg_temp"],
    "Temporal Lag":         ["dc_lag_1", "dc_lag_3", "cumulative_dc"],
}

TARGET = "has_new_dc"

def make_splits(df: pd.DataFrame):
    # Train on 2000-2018; test on 2019-2022. (Around 80/20)
    train = df[df["Year"] < 2019].copy()
    test  = df[df["Year"] >= 2019].copy()
    return train, test


def scale_features(X_train, X_test, cols):
    scaler = StandardScaler()
    Xs_train = X_train.copy()
    Xs_test  = X_test.copy()
    Xs_train[cols] = scaler.fit_transform(X_train[cols])
    Xs_test[cols]  = scaler.transform(X_test[cols])
    return Xs_train, Xs_test, scaler

def train_models(X_train, y_train):
    cw = "balanced"

    lr = LogisticRegression(
        max_iter=1000, class_weight=cw, solver="lbfgs", C=1.0, random_state=42
    )
    lr.fit(X_train, y_train)

    rf = RandomForestClassifier(
        n_estimators=300, max_depth=12, class_weight=cw,
        n_jobs=-1, random_state=42
    )
    rf.fit(X_train, y_train)

    pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        scale_pos_weight=pos_weight, subsample=0.8,
        colsample_bytree=0.8, eval_metric="logloss",
        base_score=0.5,
        random_state=42, verbosity=0,
    )
    xgb_model.fit(X_train, y_train)

    return {"Logistic Regression": lr, "Random Forest": rf, "XGBoost": xgb_model}

def evaluate(models, X_test, y_test):
    results = {}
    for name, model in models.items():
        proba = model.predict_proba(X_test)[:, 1]
        pred  = model.predict(X_test)

        precision, recall, _ = precision_recall_curve(y_test, proba)
        pr_auc = auc(recall, precision)
        f1     = f1_score(y_test, pred)
        ll     = log_loss(y_test, proba)
        roc    = roc_auc_score(y_test, proba)

        results[name] = dict(
            proba=proba, pred=pred,
            precision=precision, recall=recall,
            pr_auc=pr_auc, f1=f1, log_loss=ll, roc_auc=roc
        )
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        print(f"  PR-AUC:   {pr_auc:.4f}")
        print(f"  F1:       {f1:.4f}")
        print(f"  ROC-AUC:  {roc:.4f}")
        print(f"  Log-Loss: {ll:.4f}")
        print(classification_report(y_test, pred, target_names=["No DC", "New DC"]))

    return results

def logistic_importance(model, feature_names):
    coefs = model.coef_[0]
    return pd.Series(np.abs(coefs), index=feature_names).sort_values(ascending=False)

def gini_importance(model, feature_names):
    return pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=False)

def compute_shap(model, X_sample, model_name):
    if "XGBoost" in model_name:
        dmat = xgb.DMatrix(X_sample)
        shap_vals = model.get_booster().predict(dmat, pred_contribs=True)[:, :-1]
    else:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X_sample)
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        shap_vals = np.array(shap_vals)
        if shap_vals.ndim == 3: 
            shap_vals = shap_vals[:, :, 1]
    return np.array(shap_vals)

def ablation(X_train, y_train, X_test, y_test, feature_groups):
    all_cols_flat = [c for grp in feature_groups.values() for c in grp]
    pos_w = (y_train == 0).sum() / (y_train == 1).sum()

    def make_rf():
        return RandomForestClassifier(n_estimators=200, max_depth=10,
                                      class_weight="balanced", n_jobs=-1, random_state=42)

    def make_xgb():
        return xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                                  scale_pos_weight=pos_w, subsample=0.8, base_score=0.5,
                                  colsample_bytree=0.8, random_state=42, verbosity=0)

    print("\n  Training ablation baselines...")
    rf_base  = make_rf();  rf_base.fit(X_train[all_cols_flat],  y_train)
    xgb_base = make_xgb(); xgb_base.fit(X_train[all_cols_flat], y_train)
    baseline = {
        "Random Forest": log_loss(y_test, rf_base.predict_proba(X_test[all_cols_flat])[:, 1]),
        "XGBoost":       log_loss(y_test, xgb_base.predict_proba(X_test[all_cols_flat])[:, 1]),
    }

    print("\n\nAblation Study (ΔLog-Loss when feature group is removed):")
    print(f"{'Group':<25} {'RF':>10} {'XGB':>10}")
    print("-" * 47)

    ablation_rows = []
    for group_name, drop_cols in feature_groups.items():
        keep_cols = [c for c in all_cols_flat if c not in drop_cols]
        deltas = {}
        for name, make_fn in [("Random Forest", make_rf), ("XGBoost", make_xgb)]:
            m2 = make_fn()
            m2.fit(X_train[keep_cols], y_train)
            delta = log_loss(y_test, m2.predict_proba(X_test[keep_cols])[:, 1]) - baseline[name]
            deltas[name] = delta

        ablation_rows.append({"Group": group_name, **deltas})
        print(f"{group_name:<25} {deltas['Random Forest']:>+10.4f} {deltas['XGBoost']:>+10.4f}")

    return pd.DataFrame(ablation_rows)

def plot_pr_curves(results, save_path="pr_curves.png"):
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Logistic Regression": "#e74c3c", "Random Forest": "#2ecc71", "XGBoost": "#3498db"}
    for name, r in results.items():
        ax.plot(r["recall"], r["precision"],
                label=f"{name}  (PR-AUC={r['pr_auc']:.3f})",
                color=colors[name], linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves (2020–2023 Test Set)")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_lr_coefficients(lr_model, feature_names, save_path="lr_coefficients.png"):
    coefs = pd.Series(lr_model.coef_[0], index=feature_names).sort_values()
    colors = ["#e74c3c" if v > 0 else "#3498db" for v in coefs]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.barh(coefs.index, coefs.values, color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Standardized Coefficient")
    ax.set_title("Logistic Regression: Feature Coefficients")
    red_patch   = mpatches.Patch(color="#e74c3c", label="Increases probability")
    blue_patch  = mpatches.Patch(color="#3498db", label="Decreases probability")
    ax.legend(handles=[red_patch, blue_patch], loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_gini_importance(rf_model, xgb_model, feature_names, save_path="gini_importance.png"):
    rf_imp  = pd.Series(rf_model.feature_importances_,  index=feature_names)
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_names)

    order = rf_imp.sort_values(ascending=True).index

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (imp, title, color) in zip(
        axes,
        [(rf_imp, "Random Forest (Gini)", "#2ecc71"),
         (xgb_imp, "XGBoost (Gain)", "#3498db")]
    ):
        ax.barh(order, imp[order], color=color, edgecolor="white")
        ax.set_xlabel("Importance Score")
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.3)

    fig.suptitle("Tree-Based Feature Importance", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_shap(shap_vals, X_sample, model_name, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_vals, X_sample, show=False, plot_size=None)
    plt.title(f"SHAP Summary: {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved: {save_path}")

def plot_ablation(ablation_df, save_path="ablation.png"):
    df = ablation_df.set_index("Group")
    model_cols = [c for c in df.columns]
    x = np.arange(len(df))
    width = 0.35
    colors = {"Random Forest": "#2ecc71", "XGBoost": "#3498db"}

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(model_cols):
        ax.bar(x + j * width, df[name], width, label=name,
               color=colors.get(name, "#e74c3c"), alpha=0.85)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(df.index, rotation=20, ha="right")
    ax.set_ylabel("ΔLog-Loss (higher = group more important)")
    ax.set_title("Feature Ablation Study: Impact on Log-Loss")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Saved: {save_path}")

def plot_choropleth(df_test, test_preds, save_path="choropleth.html"):
    import plotly.express as px
    import plotly.graph_objects as go

    fips_codes = df_test["FIPS"].values

    map_df = pd.DataFrame({
        "FIPS": [str(f).zfill(5) for f in fips_codes],
        "prob": test_preds,
    }).groupby("FIPS")["prob"].mean().reset_index()

    fig = px.choropleth(
        map_df,
        geojson="https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json",
        locations="FIPS",
        color="prob",
        color_continuous_scale="YlOrRd",
        range_color=(0, map_df["prob"].quantile(0.98)),
        scope="usa",
        labels={"prob": "Predicted Prob."},
        title="Predicted Data Center Growth Likelihood (2020–2023)",
    )
    fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
    fig.write_html(save_path)
    print(f"Saved: {save_path}")

def subgroup_eval(models, df_test, y_test, best_model_name):
    model = models[best_model_name]
    pop_median = df_test["Population"].median()
    large_mask = (df_test["Population"] >= pop_median).values
    small_mask  = ~large_mask

    X_t = df_test[FEATURE_COLS]
    proba = model.predict_proba(X_t)[:, 1]
    pred  = model.predict(X_t)

    print(f"\n\nSubgroup Evaluation ({best_model_name}):")
    print(f"{'Subset':<18} {'PR-AUC':>8} {'F1':>8} {'N':>8} {'Pos%':>8}")
    print("-" * 52)

    for label, mask in [("Large counties", large_mask), ("Small counties", small_mask)]:
        yt = y_test[mask]
        pp = proba[mask]
        pd_ = pred[mask]
        if yt.sum() < 5:
            continue
        prec, rec, _ = precision_recall_curve(yt, pp)
        print(
            f"{label:<18} {auc(rec,prec):>8.4f} {f1_score(yt,pd_):>8.4f} "
            f"{len(yt):>8} {yt.mean()*100:>7.1f}%"
        )

def main():
    data_path = os.path.join("data", "county_year_dataset.csv")
    print(f"\n[1/7] Loading county-year dataset from {data_path}...")
    df = pd.read_csv(data_path, dtype={"FIPS": str, "state_fips": str})
    df["FIPS"] = df["FIPS"].str.zfill(5)
    df["state_fips"] = df["state_fips"].str.zfill(2)

    print(f"  Total rows:      {len(df):,}")
    print(f"  Counties:        {df['FIPS'].nunique():,}")
    print(f"  Pos rate (all):  {df[TARGET].mean()*100:.2f}%")
    print(f"  Total DC events: {df[TARGET].sum():,}")
    print(f"  Counties w/ DC:  {df[df[TARGET]==1]['FIPS'].nunique():,}")

    print("\n[2/7] Splitting train (<2019) / test (2019-2022)...")
    train_df, test_df = make_splits(df)
    print(f"  Train: {len(train_df):,} rows | pos={train_df[TARGET].mean()*100:.2f}%")
    print(f"  Test:  {len(test_df):,}  rows | pos={test_df[TARGET].mean()*100:.2f}%")

    X_train_raw = train_df[FEATURE_COLS]
    X_test_raw  = test_df[FEATURE_COLS]
    y_train = train_df[TARGET].values
    y_test  = test_df[TARGET].values

    numeric_cols = [c for c in FEATURE_COLS if c != "tax_exempt"]
    X_train_s, X_test_s, scaler = scale_features(X_train_raw, X_test_raw, numeric_cols)

    print("\n[3/7] Training models...")
    models_scaled = train_models(X_train_s, y_train)
    models_unscaled = train_models(X_train_raw, y_train)

    models = {
        "Logistic Regression": models_scaled["Logistic Regression"],
        "Random Forest":       models_unscaled["Random Forest"],
        "XGBoost":             models_unscaled["XGBoost"],
    }

    # Eval
    print("\n[4/7] Evaluating on test set...")
    X_test_lr  = X_test_s
    X_test_tree = X_test_raw

    results_lr  = evaluate({"Logistic Regression": models["Logistic Regression"]}, X_test_lr,   y_test)
    results_tree = evaluate(
        {"Random Forest": models["Random Forest"], "XGBoost": models["XGBoost"]},
        X_test_tree, y_test
    )
    results = {**results_lr, **results_tree}

    # Summary table
    print("\n\nSummary Table:")
    print(f"{'Model':<22} {'PR-AUC':>8} {'F1':>8} {'ROC-AUC':>9} {'Log-Loss':>10}")
    print("-" * 60)
    for name, r in results.items():
        print(f"{name:<22} {r['pr_auc']:>8.4f} {r['f1']:>8.4f} {r['roc_auc']:>9.4f} {r['log_loss']:>10.4f}")

    best_name = max(results, key=lambda n: results[n]["pr_auc"])
    print(f"\n  Best model (PR-AUC): {best_name}")

    # Feature examination
    print("\n[5/7] Feature importance...")
    lr_imp = logistic_importance(models["Logistic Regression"], FEATURE_COLS)
    print("\n  Logistic Regression (|coef|):")
    print(lr_imp.to_string())

    rf_imp  = gini_importance(models["Random Forest"], FEATURE_COLS)
    xgb_imp = gini_importance(models["XGBoost"],      FEATURE_COLS)
    print("\n  Random Forest (Gini):")
    print(rf_imp.to_string())
    print("\n  XGBoost (Gain):")
    print(xgb_imp.to_string())

    # SHAP (on 2000-sample subset for speed)
    print("\n  Computing SHAP values (Random Forest)...")
    sample_idx = np.random.choice(len(X_test_raw), min(2000, len(X_test_raw)), replace=False)
    X_shap_rf  = X_test_raw.iloc[sample_idx].reset_index(drop=True)
    shap_rf    = compute_shap(models["Random Forest"],  X_shap_rf,  "Random Forest")

    print("  Computing SHAP values (XGBoost)...")
    X_shap_xgb = X_test_raw.iloc[sample_idx].reset_index(drop=True)
    shap_xgb   = compute_shap(models["XGBoost"], X_shap_xgb, "XGBoost")

    shap_rf_mean  = pd.Series(np.abs(shap_rf).mean(axis=0),  index=FEATURE_COLS).sort_values(ascending=False)
    shap_xgb_mean = pd.Series(np.abs(shap_xgb).mean(axis=0), index=FEATURE_COLS).sort_values(ascending=False)
    print("\n  SHAP mean |values| — Random Forest:")
    print(shap_rf_mean.to_string())
    print("\n  SHAP mean |values| — XGBoost:")
    print(shap_xgb_mean.to_string())

    print("\n[6/7] Running ablation study...")
    abl_df = ablation(X_train_raw, y_train, X_test_raw, y_test, FEATURE_GROUPS)

    subgroup_eval(
        {"Random Forest": models["Random Forest"],
         "XGBoost":       models["XGBoost"]},
        test_df, y_test,
        "Random Forest"
    )

    # Create graphs
    print("\n[7/7] Generating plots...")

    all_results = {}
    for name in ["Logistic Regression", "Random Forest", "XGBoost"]:
        X_eval = X_test_lr if name == "Logistic Regression" else X_test_tree
        proba = models[name].predict_proba(X_eval)[:, 1]
        pred  = models[name].predict(X_eval)
        prec, rec, _ = precision_recall_curve(y_test, proba)
        all_results[name] = dict(
            proba=proba, pred=pred, precision=prec, recall=rec,
            pr_auc=auc(rec, prec), f1=f1_score(y_test, pred),
            roc_auc=roc_auc_score(y_test, proba),
            log_loss=log_loss(y_test, proba),
        )

    plot_pr_curves(all_results, "pr_curves.png")
    plot_lr_coefficients(models["Logistic Regression"], FEATURE_COLS, "lr_coefficients.png")
    plot_gini_importance(models["Random Forest"], models["XGBoost"], FEATURE_COLS, "gini_importance.png")

    plot_shap(shap_rf,  X_shap_rf,  "Random Forest", "shap_rf.png")
    plot_shap(shap_xgb, X_shap_xgb, "XGBoost",       "shap_xgb.png")

    plot_ablation(abl_df, "ablation.png")

    best_proba = all_results[best_name]["proba"]
    plot_choropleth(test_df, best_proba, "choropleth.html")

    print("\n" + "=" * 60)
    print("  Pipeline complete. Key outputs:")
    print("   county_year_dataset.csv — full panel")
    print("   pr_curves.png           — precision-recall curves")
    print("   lr_coefficients.png     — LR feature coefficients")
    print("   gini_importance.png     — RF + XGB Gini importance")
    print("   shap_rf.png             — SHAP summary (RF)")
    print("   shap_xgb.png            — SHAP summary (XGBoost)")
    print("   ablation.png            — feature ablation study")
    print("   choropleth.html         — interactive county map")
    print("=" * 60)

    return all_results, abl_df


if __name__ == "__main__":
    import os
    os.chdir("/Users/jasonchen/repos/repos2026/cs4641-datacenter")
    main()
