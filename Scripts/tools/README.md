# Developer Tools

These scripts are **not part of the automated pipeline**. They are standalone tools meant to be run manually by developers during debugging, evaluation, or experimentation.

None of these files are imported by the pipeline. Moving or editing them has no effect on `run_pipeline.py`.

---

## Tools

### `inspect_data.py` — Data Audit

**When to run**: Whenever you suspect a data distribution issue — outliers, scaling problems, train/val mismatch.

**What it does**: Loads all processed `.npy` sequence files from `processed_sequences/` and reports shapes, NaN/Inf counts, min/max/mean/std for each feature, outlier counts (values > 10 std), and a side-by-side comparison of train vs val distributions.

**Why it exists**: This tool was used to diagnose the `val_loss = 84.9` anomaly (see KD-13/14/15 in `docs/project-context.md`). It identified that `combined_cn_n` in the val set had a max of 1,411 scaled units (vs 90 in train), which caused a single MSE contribution of ~2.4M. Running this tool before blaming the model architecture saved hours of debugging.

```powershell
# Run from Scripts/ directory
python tools/inspect_data.py
```

Output is printed to stdout. Redirect to file if you want to save it:
```powershell
python tools/inspect_data.py > tools/inspect_data_output.txt
```

---

### `calculate_R2.py` — R² Evaluation

**When to run**: After step3 completes training and saves `test_predictions.csv`.

**What it does**: Loads test set predictions, computes per-feature and overall R² and MAE. Prints an interpretation of whether the model explains enough variance.

**Why it exists**: The training loop reports loss and MAE during training, but R² gives a more interpretable measure of prediction quality for regression tasks. Particularly useful for comparing across training runs.

```powershell
python tools/calculate_R2.py
```

> Note: Requires `Scripts/trained_model/test_predictions.csv` — must run step3 first.

---

### `train_val_test_graph.py` — Training Curve Figure

**When to run**: After training, to generate a publication-quality figure showing train/val/test MAE evolution.

**What it does**: Reads `training_history.csv` and `model_config.json`, generates a two-panel figure (overall MAE + Pc-specific MAE), and saves it to `figures/`.

**Why it exists**: step5_visualize.py generates a simpler training loss figure. This tool produces a more detailed version with test set performance overlaid at the best epoch — useful for the research report.

```powershell
python tools/train_val_test_graph.py
```

Output: `Scripts/figures/Figure_8_Train_Val_Test_Curves.png`

---

### `visualize_model_architecture.py` — Model Diagram

**When to run**: Once, when generating documentation or presentation materials.

**What it does**: Uses Keras's model visualization utilities to generate a diagram of the BiGRU architecture layers.

**Why it exists**: Used to generate architecture figures for the research report and README.

```powershell
python tools/visualize_model_architecture.py
```

---

## Important Notes

1. All tools assume they are run from the **`Scripts/` directory** — they use relative paths to `trained_model/`, `processed_sequences/`, and `figures/`.
2. Do NOT import these files from pipeline scripts. They are standalone.
3. These tools do NOT affect model weights, scaler artifacts, or any pipeline output.
