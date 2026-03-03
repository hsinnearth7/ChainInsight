# Reproducibility Protocol

> Reference: Pineau et al., "ML Reproducibility Checklist", NeurIPS 2019

## 1. Environment Lockfile

```bash
# Exact dependencies
pip install -e ".[dev]"
pip freeze > requirements.lock

# Docker image ensures identical environment
docker compose build   # → deterministic image hash
docker compose up -d
```

## 2. Randomness Control

| Source | Seed Mechanism | Notes |
|--------|----------------|-------|
| Python `random` | `random.seed(42)` | Hash-based sampling |
| NumPy | `np.random.seed(42)` | Array operations |
| PyTorch | `torch.manual_seed(42)` | Neural network init + training |
| CUDA | `torch.cuda.manual_seed_all(42)` | GPU reproducibility |
| cuDNN | `deterministic=True, benchmark=False` | Slight perf cost |
| Python hash | `PYTHONHASHSEED=42` | Dict ordering |
| LightGBM | `nthread=1` in CI | Multi-thread causes cross-platform variance |
| Float precision | Use `float64` or tolerate `ε=1e-6` | x86 vs ARM variance |

All seeds set via `app.seed.set_global_seed(42)` at startup.

## 3. Data Version Control

Synthetic data is deterministic given:
- Generator seed = 42
- Config: `configs/chaininsight.yaml` (data section)
- Generator: `app/forecasting/data_generator.py`

Verification:
```bash
python -m app.forecasting.data_generator --validate-only
# Generates data + prints SHA-256 hash
# Expected: <hash stored in configs/data_hashes.yaml>
```

## 4. Experiment Tracking

Each experiment run logs:
- Full hyperparameter configuration (from YAML)
- Random seed
- Training/validation metrics per fold
- Model artifacts (serialized)
- Evaluation results with confidence intervals

## 5. One-Command Reproduce

```bash
git clone <repo>
docker compose up -d
python -m app.reproduce --seed 42
# → runs all experiments
# → compares result hashes against expected
# → generates benchmark tables
```

## 6. Known Sources of Non-Determinism

| Source | Mitigation |
|--------|------------|
| LightGBM multi-thread | CI uses `nthread=1`; local runs may differ by `ε<0.01%` |
| Float32 platform variance | All computations use float64 |
| CUDA non-determinism | `torch.use_deterministic_algorithms(True)` when available |
| Library version drift | `requirements.lock` pins exact versions |

## 7. Validation Tolerance

Results are considered reproducible if:
- MAPE difference < 0.1% absolute
- Statistical test p-values agree on significance at α=0.05
- RL cumulative reward within 2% of reference
