# finetune_chronos.py
# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning pipeline for Chronos-2 on our SPY dataset.
#
# Compares three Chronos-2 configurations:
#   1. zero-shot            — no training, just pretrained weights
#   2. LoRA fine-tuned      — trains small adapter matrices (fast on CPU)
#   3. (optional) full FT   — retrains all 120M weights (slow, commented-out
#                             by default; flip ENABLE_FULL_FT to True to run)
#
# LoRA is the pragmatic choice on CPU: a few hundred steps, ~5-15 min,
# and typically closes most of the gap between zero-shot and full FT.
#
# Run from project root:
#     python -m src.activity1.pipelines.finetune_chronos
# ─────────────────────────────────────────────────────────────────────────────

from __future__ import annotations
import os
import time

from ..config import FIGURES_DIR
from ..data.loader import load_splits
from ..evaluation.metrics import compute_metrics, metrics_table
from ..evaluation.plots import plot_actual_vs_predicted
from ..models.chronos import Chronos2Forecaster, FineTunedChronos2Forecaster


MODELS_DIR = os.path.join(FIGURES_DIR, "models")

ENABLE_FULL_FT = False   # set True if you want to try full fine-tuning (slow)


def main() -> None:
    print("=" * 60)
    print("Activity 1 — Chronos-2 fine-tuning")
    print("=" * 60)

    splits = load_splits()
    train, val, test = splits.train, splits.val, splits.test
    print(f"\n  Loaded splits:  train={len(train)}  val={len(val)}  test={len(test)}")

    models: list = [
        Chronos2Forecaster(past_covariates=("TNX_chg",)),
        FineTunedChronos2Forecaster(
            past_covariates=("TNX_chg",),
            finetune_mode="lora",
            learning_rate=1e-5,
            num_steps=200,
            finetune_batch_size=8,
        ),
    ]
    if ENABLE_FULL_FT:
        models.append(FineTunedChronos2Forecaster(
            past_covariates=("TNX_chg",),
            finetune_mode="full",
            learning_rate=1e-6,
            num_steps=100,          # keep small on CPU
            finetune_batch_size=4,
        ))

    val_preds:  dict = {}
    test_preds: dict = {}
    results:    dict = {}

    for model in models:
        print(f"\n  ── {model.name} ──")
        t0 = time.time()
        model.fit(train)
        print(f"    fit done in {time.time()-t0:.1f}s")

        t0 = time.time()
        val_pred  = model.predict(val)
        t_val = time.time() - t0

        model.observe(val)              # extend history before test

        t0 = time.time()
        test_pred = model.predict(test)
        t_test = time.time() - t0

        val_preds[model.name]  = val_pred
        test_preds[model.name] = test_pred

        results[model.name] = {
            "val":  compute_metrics(val["target"],  val_pred),
            "test": compute_metrics(test["target"], test_pred),
        }
        print(f"    val {t_val:5.1f}s | test {t_test:5.1f}s")

    table = metrics_table(results)
    os.makedirs(MODELS_DIR, exist_ok=True)
    csv_path = os.path.join(MODELS_DIR, "chronos_finetune_metrics.csv")
    table.to_csv(csv_path)

    print("\n  Metrics:")
    print(table.round(6).to_string())
    print(f"\n  Saved: {csv_path}")

    plot_actual_vs_predicted(
        y_true=val["target"], preds=val_preds,
        title="Activity 1 — Chronos-2 (zero-shot vs LoRA) on VALIDATION",
        out_path=os.path.join(MODELS_DIR, "chronos_finetune_val.png"),
    )
    plot_actual_vs_predicted(
        y_true=test["target"], preds=test_preds,
        title="Activity 1 — Chronos-2 (zero-shot vs LoRA) on TEST",
        out_path=os.path.join(MODELS_DIR, "chronos_finetune_test.png"),
    )

    print("\n  Done.")


if __name__ == "__main__":
    main()
