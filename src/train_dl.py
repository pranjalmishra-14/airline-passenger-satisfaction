"""
Deep learning: three feed-forward architectures for tabular data.

Deliberately MLP-only. There is no spatial or temporal structure in this dataset,
so CNN/LSTM/transformer architectures would add complexity without a defensible
rationale. Layer sizes are derived from the actual encoded feature count rather
than copied from an arbitrary reference architecture.

The validation split drives EarlyStopping and ModelCheckpoint; the test set is
never seen during training.
"""
from __future__ import annotations

import time
import traceback

import joblib
import numpy as np

import config
from src import evaluate as ev
from src import utils
from src.preprocessing import build_preprocessor


def _keras():
    import keras
    return keras


def build_basic_mlp(n_features: int):
    """Model A -- baseline MLP: Dense -> BN -> ReLU -> Dropout, twice."""
    keras = _keras()
    h1 = max(64, min(256, n_features * 8))
    h2 = h1 // 2
    m = keras.Sequential([
        keras.layers.Input((n_features,)),
        keras.layers.Dense(h1), keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"), keras.layers.Dropout(0.3),
        keras.layers.Dense(h2), keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"), keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="basic_mlp")
    return m


def build_regularized_mlp(n_features: int):
    """Model B -- stronger regularisation: L2 penalties + higher dropout."""
    keras = _keras()
    reg = keras.regularizers.l2(1e-4)
    h1 = max(64, min(256, n_features * 8))
    h2 = h1 // 2
    m = keras.Sequential([
        keras.layers.Input((n_features,)),
        keras.layers.Dense(h1, kernel_regularizer=reg), keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"), keras.layers.Dropout(0.45),
        keras.layers.Dense(h2, kernel_regularizer=reg), keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"), keras.layers.Dropout(0.45),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="regularized_mlp")
    return m


def build_deep_mlp(n_features: int):
    """Model C -- deeper network: four hidden blocks with a tapering width."""
    keras = _keras()
    h = max(128, min(384, n_features * 12))
    m = keras.Sequential([
        keras.layers.Input((n_features,)),
        keras.layers.Dense(h), keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"), keras.layers.Dropout(0.3),
        keras.layers.Dense(h // 2), keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"), keras.layers.Dropout(0.3),
        keras.layers.Dense(h // 4), keras.layers.BatchNormalization(),
        keras.layers.Activation("relu"), keras.layers.Dropout(0.2),
        keras.layers.Dense(h // 8), keras.layers.Activation("relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="deep_mlp")
    return m


DL_ARCHITECTURES = {
    "Neural Net (Basic MLP)": (build_basic_mlp, "Two hidden blocks, dropout 0.3."),
    "Neural Net (Regularized MLP)": (build_regularized_mlp, "L2 1e-4 + dropout 0.45."),
    "Neural Net (Deep MLP)": (build_deep_mlp, "Four hidden blocks, tapering width."),
}


def _slug(name: str) -> str:
    return (name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            .replace("-", "_"))


def train_one_dl(name: str, builder, note: str, Xtr, ytr, Xva, yva, budgets: dict, manifest):
    """Compile, train with early stopping, and evaluate one Keras model."""
    keras = _keras()
    n_features = Xtr.shape[1]
    model = builder(n_features)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    ckpt = config.MODELS_DIR / f"{_slug(name)}.keras"
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True, verbose=0,
        ),
        keras.callbacks.ModelCheckpoint(
            str(ckpt), monitor="val_loss", save_best_only=True, verbose=0,
        ),
    ]

    t0 = time.perf_counter()
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),      # validation set -- never the test set
        epochs=budgets["epochs"],
        batch_size=config.BATCH_SIZE,
        callbacks=callbacks,
        verbose=0,
    )
    fit_seconds = time.perf_counter() - t0
    epochs_run = len(history.history["loss"])

    t1 = time.perf_counter()
    y_score = model.predict(Xva, verbose=0).ravel()
    predict_seconds = time.perf_counter() - t1
    y_pred = (y_score >= 0.5).astype(int)
    metrics = ev.compute_metrics(yva, y_pred, y_score)

    manifest.record_model(
        name,
        family="deep",
        fit_seconds=round(fit_seconds, 2),
        predict_seconds=round(predict_seconds, 2),
        epochs_run=epochs_run,
        epochs_budget=budgets["epochs"],
        parameters=int(model.count_params()),
        architecture=note,
        val_metrics={k: round(float(v), 4) for k, v in metrics.items()},
        saved_to=str(ckpt.relative_to(config.ROOT)),
        status="ok",
    )
    utils.info(
        f"val acc {metrics['Accuracy']:.4f} | F1 {metrics['F1']:.4f} | "
        f"AUC {metrics['ROC-AUC']:.4f} | {epochs_run} epochs | {fit_seconds:.1f}s"
    )

    row = {
        "Model": name,
        "Family": "deep",
        **metrics,
        "Fit (s)": round(fit_seconds, 2),
        "Predict (s)": round(predict_seconds, 2),
        "Epochs": epochs_run,
        "Parameters": int(model.count_params()),
        "Status": "ok",
        "Note": note,
    }
    return row, history.history, model


def train_all_dl(splits, mode: str, manifest):
    """
    Train all three MLPs.

    The preprocessor is fitted on train only and saved so the app can reuse the
    exact same transformation at inference time.
    """
    budgets = config.MODE_BUDGETS[mode]
    utils.set_seeds()

    pre = build_preprocessor(splits.X_train, scale=True)
    Xtr = pre.fit_transform(splits.X_train).astype("float32")
    Xva = pre.transform(splits.X_val).astype("float32")
    ytr = splits.y_train.to_numpy().astype("float32")
    yva = splits.y_val.to_numpy().astype("float32")

    joblib.dump(pre, config.MODELS_DIR / "dl_preprocessor.pkl", compress=3)

    rows, histories, models = [], {}, {}
    for name, (builder, note) in DL_ARCHITECTURES.items():
        utils.step(f"Training {name}...")
        try:
            row, hist, model = train_one_dl(
                name, builder, note, Xtr, ytr, Xva, yva, budgets, manifest
            )
            rows.append(row)
            histories[name] = hist
            models[name] = model
        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            utils.warn(f"{name} FAILED: {type(exc).__name__}: {exc}")
            manifest.record_failure(name, f"{type(exc).__name__}: {exc}", tb)
            rows.append({
                "Model": name, "Family": "deep",
                "Status": f"FAILED: {type(exc).__name__}: {exc}", "Note": note,
            })

    return rows, histories, models, pre
