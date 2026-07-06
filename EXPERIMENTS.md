# Experiments & Entry-Point Map

This document maps every experiment in this repository to the code that actually
runs it, classifies the non-obvious files (the `*_debug` / `*_validation` /
`_smaller` variants), and records correctness issues found while mapping.

> **Why this exists:** filenames here are misleading. Some files that look like
> throwaway "debug" scripts are (or were) real runners, and some clean-looking
> files are broken as committed. Decisions about renaming/removing/refactoring
> must be based on this map, not on filenames.

---

## 1. How an experiment is launched

There is no direct `python attack.py` in normal use. The chain is:

1. A thin launcher in `source/sweeps/*.py` calls
   `run_sweep(entity, project, sweep_config_path)`
   ([source/utils/wandb_helpers.py:98](source/utils/wandb_helpers.py#L98)).
2. `run_sweep` reads the YAML, calls `wandb.sweep(...)`, then shells out:
   `os.system("nice -n 0 wandb agent <entity>/<project>/<sweep_id>")`.
3. The wandb agent runs the script named in the sweep config's **`program:`**
   field as `python <program>`.

**Therefore the `program:` field of each `sweep_configs/*.yaml` is the true
entry point.** The `*_sweep.py` launchers do **not** import attack/defense code.

**Config injection — two styles (this is the key tell for canonical vs variant):**
- **Canonical runners** read hyperparameters from the flat `wandb.config` object
  the agent injects (e.g. `config.layer_size`).
- **Debug/legacy variants** instead load a YAML file with
  `load_config_file(...)` and read the *first* value of each list:
  `config.parameters['layer_size']['values'][0]`. These run a single
  configuration locally without a real sweep.

Sweeps run **sequentially** (one `wandb agent` process walks the whole grid) —
this is the target for the later parallelization work.

---

## 2. Canonical entry-point map

| Experiment | Sweep config | `program:` → file | Config style | Status |
|---|---|---|---|---|
| Base-model training | `Adult_sweep_config.yaml` | `source/training/train_adult.py` | `wandb.config` | ✅ live, import-clean |
| Hyperparameter grid search | `GridSearch_adult_sweep_config.yaml` | `source/training/hyperparam_tuning_adult.py` | `wandb.config` | ✅ live (runs at import; no `__main__` guard) |
| LSB attack (+ defense) | `LSB_sweep_config.yaml`, `LSB_adult_sweep/config.yaml` | `source/attacks/LSB_attack.py` | `wandb.config` | ✅ live, import-clean |
| Sign-Encoding (SE) attack | `SE_sweep_config.yaml`, `Sign_Encoding_sweep/config.yaml` | `source/attacks/SE_attack.py` | `wandb.config` | ✅ live, import-clean |
| Correlated-Value-Encoding (CVE) attack | `CVE_sweep_config.yaml`, `CorrVal_Encoding_sweep/config.yaml` | `source/attacks/CVE_attack.py` → **`cve_attack.py`** | mixed ⚠️ | ⚠️ live but fragile (see §4.2) |
| SE defense (sign modification) | `SE_defense_sweep_config.yaml` | `source/defenses/sign_modification_defense.py` | `wandb.config` | ❌ import-broken as committed (see §4.1) |
| CVE defense (parameter rotation) | `CVE_defense_sweep_config.yaml` | `source/defenses/parameter_rotation_defense.py` | `wandb.config` | ❌ import-broken as committed (see §4.1) |
| Black-box attack | `BB_sweep_config.yaml` | `source/attacks/BB_attack.py` | `wandb.config` | ✅ live, import-clean |
| Black-box defense (pruning) | `BB_defense_sweep_config.yaml` | `source/defenses/black_box_defense.py` | `wandb.config` | ❌ import-broken as committed (see §4.1) |

Data prerequisite: the runners read pre-generated `tabular_data/adult_data_*.csv`
files produced by **`source/training/save_prep_data.py`** (see §5).

---

## 3. Per-experiment detail

### 3.1 Base-model training — `source/training/train_adult.py`
Trains the "clean" base MLPs the attacks/defenses later operate on. Sweep-driven
(`wandb.init()` + `wandb.config`, [train_adult.py:268](source/training/train_adult.py#L268)).
Single-pass training on the full pre-split train set (no CV — deliberate, for
comparability across sizes). Saves to
`models/{dataset}/base_models/{nhl}hl_{ls}s/model.pth`. This is the "model zoo"
producer.

### 3.2 Hyperparameter search — `source/training/hyperparam_tuning_adult.py`
5-fold StratifiedKFold CV with early stopping; how the base architecture/HPs were
chosen. Runs entirely at module import (no `__main__` guard). `dataset` is
hardcoded to `"adult"` (ignores `config.dataset`). Hardcodes wandb
`entity='siposova-andrea'` ([hyperparam_tuning_adult.py:22](source/training/hyperparam_tuning_adult.py#L22)).

### 3.3 LSB attack — `source/attacks/LSB_attack.py`
Least-significant-bit encoding of the secret into model parameters, plus the LSB
defense (bit overwrite) in the same script. Clean, sweep-driven
(`wandb.config`, [LSB_attack.py:456](source/attacks/LSB_attack.py#L456)),
`__main__` guard present. Writes results CSV + wandb.

### 3.4 SE attack — `source/attacks/SE_attack.py`
Sign-Encoding: penalty term aligns weight signs with secret bits; reconstruction
reads the signs. Canonical, sweep-driven (`attack_config = wandb.config`,
[SE_attack.py:428](source/attacks/SE_attack.py#L428)). Logs numerical/categorical
similarity separately; tracks base vs malicious correct-sign proportion. Writes
`results/{dataset}/sign_encoding_attack/*.csv` + wandb. See §4.3 for its
`se_attack_debug.py` sibling.

### 3.5 CVE attack — `source/attacks/cve_attack.py`
Correlated-Value-Encoding: adds a regularizer correlating weights with the secret
vector; reconstruction recovers it. **Config handling is inconsistent** — see
§4.2. Writes `results/{dataset}/correlated_value_encoding_attack/*.csv` + wandb.

### 3.6 SE defense — `source/defenses/sign_modification_defense.py`
Flips/overwrites the `percent_to_modify` smallest-magnitude weights to destroy the
sign-encoded payload; sweeps the modification percentage and measures
attack-degradation vs accuracy-loss. Loads the SE benign/malicious models. **Broken
import as committed** (§4.1).

### 3.7 CVE defense — `source/defenses/parameter_rotation_defense.py`
Decorrelates each linear layer via SVD singular-value flattening + QR blending
(strength-swept), renormalizing to preserve weight norm. Loads the CVE
(`corrval_encoding`) models. **Broken import as committed** (§4.1). Hardcodes the
CVE wandb project name.

### 3.8 Black-box attack — `source/attacks/BB_attack.py`
API-only attack: trains a malicious model that fits "trigger" samples whose labels
encode the private data; reconstructs from trigger predictions. Trigger features
generated by `mal_data_generation ∈ {known_d_ood, known_d_id, uniform}`, oversampled
by `repetition`, scaled with a separate scaler. Canonical, sweep-driven
(`wandb.config`, [BB_attack.py:492](source/attacks/BB_attack.py#L492)),
**import-clean** (uses migrated `source/helpers`, `source/similarity`). Saves
`models/{dataset}/black_box/{benign|malicious}/{nhl}hl_{ls}s/{ratio}ratio_{rep}rep_{gen}.pth`
+ results CSV + wandb.

### 3.9 Black-box defense — `source/defenses/black_box_defense.py`
Prunes low-average-activation neurons across a pruning sweep and measures how the
attack's reconstruction similarity and clean accuracy degrade; renders
neuron-pruning plots. Loads the BB attack's saved models (attack→defense handoff).
Sweep-driven, but **broken import as committed** (§4.1) and hardcodes
`project = "BB"`.

---

## 4. Correctness backlog (found while mapping — for the "understanding/verify" phase)

> These are **not yet fixed.** They are recorded here so the refactor addresses
> them deliberately, per-item, after the code is understood.

### 4.1 All four defense entry points fail to import as committed ❗ (verified)
`cm_class_acc` and `baseline` are defined **only** in
`source/evaluation/evaluation.py`, but:
- `sign_modification_defense.py:17` and `parameter_rotation_defense.py:17` import
  `cm_class_acc, baseline` from `source.helpers.black_box_helpers` (not there) → `ImportError`.
- `noise_injection_defense_SE.py:17` imports `baseline` from the same wrong place → `ImportError`.
- `black_box_defense.py:20-22` imports `generate_malicious_data, reconstruct_from_preds, cm_class_acc, baseline` /
  `convert_label_enc_to_binary` / `calculate_similarity` from
  `source.attacks.black_box_helpers` / `source.attacks.lsb_helpers` / `source.attacks.similarity` —
  **all three modules are missing** (now under `source/helpers/` and `source/similarity/`) → `ModuleNotFoundError`.

The *attack* files (`BB_attack.py`) import from the correct migrated locations.
Conclusion: the `34fd161 "Restructured and updated black-box files"` migration
updated the attacks but not the defenses. The committed defense results were
almost certainly generated before that restructure. **Fixing these imports is a
prerequisite to reproducing any defense experiment.**

### 4.2 CVE attack config handling + filename case ⚠️
- `cve_attack.py` mixes styles: `train()` reads `config.parameters['x']['values'][0]`
  (file/YAML style, [cve_attack.py:123](source/attacks/cve_attack.py#L123)) while
  `run_training()` reassigns `attack_config = wandb.config` (flat style,
  [cve_attack.py:470](source/attacks/cve_attack.py#L470)). Under a real
  `wandb agent`, flat `wandb.config` has no `.parameters`, so this path looks like
  it would break — needs a runtime check.
- **Case mismatch:** every CVE config says `program: source/attacks/CVE_attack.py`,
  but git only ever tracked lowercase `cve_attack.py`. This resolves on
  case-insensitive macOS but **would fail on Linux** (where the code was originally
  run — see the leftover `/home/siposova/PycharmProjects/...` paths). Yet CVE
  results exist, so how it ran needs to be understood before "fixing" the case.
  Resolution (later): standardize on one lowercase name everywhere.

### 4.3 SE debug sibling — `se_attack_debug.py`
Not the runner (see §5), but note it has **mislabeled metrics**: the base/malicious
correct-sign-proportion values are swapped between the `train_epoch` calls and the
logged keys. If it was ever used to produce any reported number, that number is
suspect.

### 4.4 Sweep-config inconsistencies
- `Sign_Encoding_sweep/config.yaml` uses a bare `program: SE_attack.py` (no
  `source/attacks/` path) — unlike the other SE config.
- `Black_box_adult_sweep/config.yaml` has **no `program:` field at all** — it
  cannot launch anything on its own.

### 4.5 Hardcoded wandb identity/projects
`Configuration.ENTITY = 'your-name'` (placeholder), but
`hyperparam_tuning_adult.py` hardcodes `entity='siposova-andrea'`,
`black_box_defense.py` hardcodes `project="BB"`, and
`parameter_rotation_defense.py` hardcodes the CVE project name — bypassing
`Configuration`.

### 4.6 Results collection is inconsistent
Attacks write a per-epoch CSV **and** log to wandb; defenses log **only** to wandb
(no CSV). Each attack logs to a **different wandb project** (`BB_PROJECT`,
`SE_PROJECT`, `CVE_PROJECT`, `LSB_PROJECT`, `PROJECT_ECC` in `Configuration`). This
is the surface the "rebuild results collection" work will unify.

---

## 5. Non-canonical / variant / dead files — do **not** delete on filename

| File | Verdict | Evidence |
|---|---|---|
| `source/attacks/se_attack_debug.py` | **Debug harness, keep.** Single-config, file-driven SE runner (`load_config_file(Sign_Encoding_sweep)`, `wandb.config` commented out). Not wired to any sweep. Has mislabeled metrics (§4.3). | Reads params as `config.parameters[...][values][0]`; saves benign at epoch 10 vs 30 in the canonical file. |
| `source/attacks/black_box_attack_validation.py` | **Superseded predecessor, dead.** CV-wrapped BB attack that also inlined the pruning defense. Not "an attack on the validation set." | Not wired anywhere (only in README tree). Git: both it and `BB_attack.py` descend from `1000623`; `BB_attack.py` continued (`839f918`→`34fd161`), this froze. Can't import (stale `source.attacks.*`), and can't finish an epoch (`base_model == "False"` string-compare bug). |
| `source/training/train_adult_smaller.py` | **Dead early prototype.** Fixed 4-hidden-layer CV trainer; precursor to `hyperparam_tuning_adult.py`. | Its `program:` line is text inside a docstring (bare filename). No sweep targets it. Bare `from data_loading import ...` breaks under the current layout. One git commit ever. |
| `source/helpers/save_prep_data.py` | **4-line stub, effectively dead.** Imports `LSB_attack.get_data_for_training`, prints `X_train`. Saves nothing. | Not imported anywhere. |
| `source/defenses/noise_injection_defense_SE.py` | **Experimental, unfinished.** Alternative SE defense: adds Gaussian noise to smallest-magnitude weights instead of flipping signs. | Wired to no config; needs a `noise_scale` key no YAML has; runner still named `run_sm_defense`; `TypeError` at line 124 (`WeightModifier(attacked_model)` missing `noise_scale`); plus the §4.1 import bug. A "finish it" candidate, not a working experiment. |

**The real data-prep entry point** is `source/training/save_prep_data.py` (27 lines):
run it (`python source/training/save_prep_data.py`) to (re)generate the six
`adult_data_*.csv` files the trainers/attacks read. Neither `save_prep_data.py` is
imported anywhere; both are run for side effects.

---

## 6. Dataset-hardcoding inventory (drives the dataset-agnostic refactor)

The whole pipeline assumes Adult. The **only** dataset-agnostic piece is
`MyDataset` in `data_loading.py`. To add datasets, these must be generalized:

- **`source/data_loading/data_loading.py`** — hardcoded raw 15-column schema and
  filenames (`adult.data`/`adult.test`); `capital_change = capital_gain - capital_loss`
  feature engineering; `native_country` binarized to US/NotUS; fixed 7 categorical
  columns; **hardcoded label map `{'<=50K':0, '>50K':1}`**; target assumed to be the
  last column.
- **Every runner** gates all logic on `if dataset == 'adult':` with **no `else`** —
  a non-adult sweep silently leaves `X_train`/`input_size`/etc. undefined
  (`NameError`).
- **`hidden_num_cols = ["age","education_num","capital_change","hours_per_week"]`**
  hardcoded across `SE_attack.py`, `cve_attack.py`, `BB_attack.py`, and the three
  defenses.
- **Confusion-matrix `class_names=["<=50K", ">50K"]`** hardcoded in every
  `wandb.plot.confusion_matrix` call.
- **`black_box_defense.py` hardcodes `input_size = 41`** (the Adult one-hot width).

---

## 7. Model / results path scheme (for reference)

- Base models: `models/{dataset}/base_models/{nhl}hl_{ls}s/model.pth`
- Attack outputs (example, BB):
  `models/{dataset}/black_box/{benign|malicious}/{nhl}hl_{ls}s/{ratio}ratio_{rep}rep_{gen}.pth`
- SE/CVE models: `models/{dataset}/{sign_encoding|corrval_encoding}/{benign|malicious}/{nhl}hl_{ls}s/penalty_{lambda_s}.pth`
- Results CSVs (attacks): `results/{dataset}/{attack_name}/*.csv`
- All roots come from `source/utils/Configuration.py`
  (`BASE_DIR = <repo root>`), so paths are portable **except** the case/identity
  issues in §4.2 and §4.5.

---

*Generated during the entry-point mapping phase of the refactor. Update this file
as the correctness items in §4 are resolved and as new datasets/methods are added.*
