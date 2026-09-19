"""
End-to-end training pipeline.

    python train.py --mode fast    # quick debugging run
    python train.py --mode full    # the real experiment

Protocol enforced here:
  * Data is split once (70/15/15) in src/data_loader.
  * All model selection uses the VALIDATION set.
  * The TEST set is opened exactly once, at the end, after the final model is chosen.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import joblib
import numpy as np
import pandas as pd

import config
from src import data_loader as dl
from src import eda as eda_mod
from src import evaluate as ev
from src import explainability as xai
from src import plots, poster
from src import train_dl, train_ml, utils
from src.registry import get_specs

warnings.filterwarnings("ignore", category=UserWarning)


def parse_args():
    p = argparse.ArgumentParser(description="Airline passenger satisfaction: ML vs DL")
    p.add_argument("--mode", choices=["fast", "full"], default="full",
                   help="fast = small subsample and tiny budgets; full = the real experiment")
    p.add_argument("--skip-eda", action="store_true", help="skip EDA figure generation")
    p.add_argument("--skip-dl", action="store_true", help="skip deep learning models")
    return p.parse_args()


def main():
    args = parse_args()
    mode = args.mode
    budgets = config.MODE_BUDGETS[mode]
    utils.set_seeds()

    n_ml = len(get_specs(mode))
    n_dl = 0 if args.skip_dl else len(train_dl.DL_ARCHITECTURES)
    total = 3 + (0 if args.skip_eda else 1) + n_ml + n_dl + 5
    utils.start_run(total)

    manifest = utils.Manifest(mode)
    print(f"\n{'=' * 78}\nAirline Passenger Satisfaction -- mode: {mode.upper()}\n{'=' * 78}\n")

    # ---------------------------------------------------------------- 1. data
    utils.step("Loading dataset...")
    raw = dl.load_raw()
    target = dl.find_target(raw)
    drops = dl.detect_droppable(raw, target)
    utils.info(f"{raw.shape[0]:,} rows x {raw.shape[1]} columns | target: '{target}'")
    utils.info(f"dropped: {drops if drops else 'nothing'}")

    utils.step("Analysing data quality...")
    report = dl.quality_report(raw, target, drops)
    (config.REPORTS_DIR / "data_quality.md").write_text(report)
    utils.info(f"report -> outputs/reports/data_quality.md")

    utils.step("Creating stratified 70/15/15 split...")
    splits = dl.make_splits(raw, sample_rows=budgets["sample_rows"])
    splits.assert_disjoint()
    utils.info(f"train {splits.sizes['train']:,} | val {splits.sizes['val']:,} | "
               f"test {splits.sizes['test']:,}")
    utils.info(f"positive-class rate -- train {splits.y_train.mean():.4f}, "
               f"val {splits.y_val.mean():.4f}, test {splits.y_test.mean():.4f}")
    manifest.record_split(
        **splits.sizes,
        fractions={"train": config.TRAIN_SIZE, "val": config.VAL_SIZE, "test": config.TEST_SIZE},
        dropped_columns=drops,
        target=target,
        n_raw_features=splits.X_train.shape[1],
        positive_label=config.POSITIVE_LABEL,
    )

    # ----------------------------------------------------------------- 2. EDA
    if not args.skip_eda:
        utils.step("Generating EDA figures...")
        eda_mod.run_eda(raw.drop(columns=list(drops)), target)

    # ------------------------------------------------- 3. classical + ensemble
    ml_rows = train_ml.train_all(splits, mode, manifest)

    # ------------------------------------------------------- 4. deep learning
    dl_rows, histories, dl_models, dl_pre = [], {}, {}, None
    if not args.skip_dl:
        dl_rows, histories, dl_models, dl_pre = train_dl.train_all_dl(splits, mode, manifest)
        if histories:
            plots.plot_training_history(histories)

    # --------------------------------------------- 5. select the final model
    utils.step("Selecting final model (validation evidence only)...")
    val_rows = [r for r in (ml_rows + dl_rows) if r.get("Status") == "ok"]
    if not val_rows:
        raise RuntimeError("No model trained successfully; cannot continue.")
    val_df = ev.results_table(val_rows)
    val_df.to_csv(config.OUTPUTS_DIR / "validation_results.csv", index=False)

    best_name = val_df.iloc[0]["Model"]
    utils.info(f"best on validation F1: {best_name} (F1={val_df.iloc[0]['F1']:.4f})")
    runner_up = val_df.iloc[1] if len(val_df) > 1 else None
    if runner_up is not None:
        gap = val_df.iloc[0]["F1"] - runner_up["F1"]
        utils.info(f"margin over runner-up ({runner_up['Model']}): {gap:.4f} F1")

    # --------------------------- 6. FINAL TEST EVALUATION -- the only test use
    utils.step("Final evaluation on the held-out TEST set (used once)...")
    test_rows, cms, curves = [], {}, {}

    for spec in get_specs(mode):
        path = config.MODELS_DIR / f"{train_ml._slug(spec.name)}.pkl"
        if not path.exists():
            continue
        pipe = joblib.load(path)
        metrics, y_pred, y_score = ev.evaluate_model(pipe, splits.X_test, splits.y_test)
        row = {"Model": spec.name, "Family": spec.family, **metrics, "Status": "ok",
               "Note": spec.note}
        # Bootstrap CIs for every model, so "is the gap real?" can be answered.
        row.update(ev.bootstrap_ci(splits.y_test, y_pred, "f1", budgets["bootstrap_n"]))
        test_rows.append(row)
        cms[spec.name] = ev.confusion_frame(splits.y_test, y_pred)
        if y_score is not None:
            curves[spec.name] = (splits.y_test.to_numpy(), y_score)

    if dl_models and dl_pre is not None:
        Xte = dl_pre.transform(splits.X_test).astype("float32")
        for name, model in dl_models.items():
            score = model.predict(Xte, verbose=0).ravel()
            pred = (score >= 0.5).astype(int)
            metrics = ev.compute_metrics(splits.y_test, pred, score)
            row = {"Model": name, "Family": "deep", **metrics, "Status": "ok",
                   "Note": train_dl.DL_ARCHITECTURES[name][1]}
            row.update(ev.bootstrap_ci(splits.y_test, pred, "f1", budgets["bootstrap_n"]))
            test_rows.append(row)
            cms[name] = ev.confusion_frame(splits.y_test, pred)
            curves[name] = (splits.y_test.to_numpy(), score)

    # Carry CV columns across from the validation stage.
    cv_cols = ["CV Acc Mean", "CV Acc Std", "CV F1 Mean", "CV F1 Std"]
    cv_map = {r["Model"]: {c: r.get(c) for c in cv_cols if c in r} for r in val_rows}
    for r in test_rows:
        r.update(cv_map.get(r["Model"], {}))

    # Record models that failed so they appear in the results table.
    for r in (ml_rows + dl_rows):
        if r.get("Status") != "ok":
            test_rows.append(r)

    results = ev.results_table(test_rows)
    results.to_csv(config.RESULTS_CSV, index=False)
    dl_only = results[results["Family"] == "deep"]
    if not dl_only.empty:
        dl_only.to_csv(config.DL_RESULTS_CSV, index=False)

    ok = results[results["Status"] == "ok"]
    final_name = ok.iloc[0]["Model"]
    utils.info(f"top model on test F1: {final_name} (F1={ok.iloc[0]['F1']:.4f}, "
               f"Acc={ok.iloc[0]['Accuracy']:.4f})")

    plots.plot_model_comparison(results)
    plots.plot_confusion_matrices(cms)
    plots.plot_roc_pr(curves)
    plots.plot_ci_forest(results)

    # -------------------------------------------------------- 7. explainability
    utils.step("Computing feature importance and SHAP...")
    ml_pipelines = {}
    for spec in get_specs(mode):
        p = config.MODELS_DIR / f"{train_ml._slug(spec.name)}.pkl"
        if p.exists():
            ml_pipelines[spec.name] = joblib.load(p)

    imp = xai.tree_feature_importance(ml_pipelines, splits.X_train)
    if not imp.empty:
        imp.to_csv(config.FEATURE_IMPORTANCE_CSV, index=False)
        focus = [m for m in ("Random Forest", "XGBoost") if m in imp["Model"].unique()]
        plots.plot_feature_importance(imp[imp["Model"].isin(focus)] if focus else imp)
        utils.info(f"feature importance -> {config.FEATURE_IMPORTANCE_CSV.name}")

    shap_summary = None
    tree_candidates = [m for m in ok["Model"] if m in ml_pipelines
                       and hasattr(ml_pipelines[m].named_steps["model"], "feature_importances_")]
    if tree_candidates:
        shap_model = tree_candidates[0]
        utils.info(f"SHAP TreeExplainer on: {shap_model}")
        sample = splits.X_test.sample(
            min(2000, len(splits.X_test)), random_state=config.RANDOM_STATE
        )
        try:
            shap_summary = xai.shap_analysis(ml_pipelines[shap_model], sample, shap_model)
            (config.SHAP_DIR / "shap_summary.json").write_text(
                json.dumps(shap_summary, indent=2, default=str)
            )
            top = ", ".join(r["Feature"] for r in shap_summary["top_features"][:5])
            utils.info(f"top SHAP features: {top}")
        except Exception as exc:  # noqa: BLE001
            utils.warn(f"SHAP failed: {type(exc).__name__}: {exc}")
            manifest.record_failure("SHAP", f"{type(exc).__name__}: {exc}")

    # Permutation importance for the best neural network (SHAP Kernel is too slow).
    if dl_models and dl_pre is not None:
        try:
            nn_name = max(
                dl_models,
                key=lambda n: float(ok.loc[ok["Model"] == n, "F1"].iloc[0])
                if (ok["Model"] == n).any() else -1,
            )
            n_rep = 1 if mode == "fast" else 3
            sub = splits.X_val.sample(
                min(3000, len(splits.X_val)), random_state=config.RANDOM_STATE
            )
            perm = xai.permutation_importance_nn(
                dl_models[nn_name], dl_pre, sub, splits.y_val.loc[sub.index], n_repeats=n_rep
            )
            perm.insert(0, "Model", nn_name)
            perm.to_csv(config.OUTPUTS_DIR / "nn_permutation_importance.csv", index=False)
            utils.info(f"NN permutation importance ({nn_name}) -> nn_permutation_importance.csv")
        except Exception as exc:  # noqa: BLE001
            utils.warn(f"permutation importance skipped: {exc}")

    # ----------------------------------------------------- 8. save final model
    utils.step("Saving final model and feature schema...")
    if final_name in ml_pipelines:
        final_kind = "sklearn"
        joblib.dump(ml_pipelines[final_name], config.FINAL_MODEL_PATH, compress=3)
        final_artifacts = [config.FINAL_MODEL_PATH.name]
    else:
        # The winner is a neural network. Keras models cannot be pickled reliably,
        # so best_model.pkl stores a pointer to the .keras file plus its
        # preprocessor; the app loads the pair. best_model.pkl is always written
        # so downstream consumers can rely on it existing.
        final_kind = "keras"
        joblib.dump(dl_pre, config.MODELS_DIR / "dl_preprocessor.pkl", compress=3)
        keras_file = config.MODELS_DIR / f"{train_dl._slug(final_name)}.keras"
        joblib.dump(
            {
                "kind": "keras",
                "model_file": keras_file.name,
                "preprocessor_file": "dl_preprocessor.pkl",
                "model_name": final_name,
            },
            config.FINAL_MODEL_PATH,
            compress=3,
        )
        final_artifacts = [config.FINAL_MODEL_PATH.name, keras_file.name,
                           "dl_preprocessor.pkl"]
    utils.info(f"final model: {final_name} ({final_kind}) -> {', '.join(final_artifacts)}")

    schema = {
        "final_model": final_name,
        "final_model_kind": final_kind,
        "target": target,
        "positive_label": config.POSITIVE_LABEL,
        "negative_label": config.NEGATIVE_LABEL,
        "raw_features": [
            {
                "name": c,
                "dtype": str(splits.X_train[c].dtype),
                "kind": "numeric" if pd.api.types.is_numeric_dtype(splits.X_train[c]) else "categorical",
                "min": float(splits.X_train[c].min()) if pd.api.types.is_numeric_dtype(splits.X_train[c]) else None,
                "max": float(splits.X_train[c].max()) if pd.api.types.is_numeric_dtype(splits.X_train[c]) else None,
                "median": float(splits.X_train[c].median()) if pd.api.types.is_numeric_dtype(splits.X_train[c]) else None,
                "categories": sorted(splits.X_train[c].dropna().unique().tolist()) if not pd.api.types.is_numeric_dtype(splits.X_train[c]) else None,
            }
            for c in splits.X_train.columns
        ],
    }
    config.SCHEMA_PATH.write_text(json.dumps(schema, indent=2))
    utils.info(f"feature schema -> models/{config.SCHEMA_PATH.name}")

    # ---------------------------------------------------------- 9. manifest
    utils.step("Building summary poster...")
    try:
        name = poster.build_poster()
        utils.info(f"poster -> outputs/figures/{name} (+ PDF in outputs/reports/)")
    except Exception as exc:  # noqa: BLE001
        utils.warn(f"poster generation skipped: {type(exc).__name__}: {exc}")

    utils.step("Writing experiment manifest...")
    manifest.data["final_model"] = final_name
    manifest.data["reference_benchmarks"] = config.REFERENCE_PAPER
    manifest.note("Test set was evaluated exactly once, after model selection on validation.")
    manifest.note(f"Tuning budget: {budgets['search_iter']} candidates x "
                  f"{budgets['search_cv']} folds (train only).")
    if shap_summary:
        manifest.data["shap"] = {"model": shap_summary["model"],
                                 "top_features": shap_summary["top_features"][:10]}
    manifest.save()
    utils.info(f"manifest -> outputs/{config.MANIFEST_JSON.name}")

    # ------------------------------------------------------------- summary
    print(f"\n{'=' * 78}\nRESULTS (held-out test set)\n{'=' * 78}")
    show = [c for c in ["Model", "Family", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
            if c in results.columns]
    print(results[show].to_string(index=False))
    if manifest.data["failures"]:
        print("\nFAILED MODELS (reported, not skipped):")
        for k, v in manifest.data["failures"].items():
            print(f"  - {k}: {v['reason']}")
    print(f"\nFinal model: {final_name}")
    print(f"Results  -> {config.RESULTS_CSV.relative_to(config.ROOT)}")
    print(f"Manifest -> {config.MANIFEST_JSON.relative_to(config.ROOT)}")
    print(f"Figures  -> {config.FIGURES_DIR.relative_to(config.ROOT)}/\n")


if __name__ == "__main__":
    main()
