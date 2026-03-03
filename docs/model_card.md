# Model Card: ChainInsight Demand Forecasting System

> Following Mitchell et al., "Model Cards for Model Reporting", FAT* 2019

## Model Details

- **Model name:** ChainInsight Routing Ensemble
- **Version:** 2.0.0
- **Type:** Time series demand forecasting (multi-model routing ensemble)
- **Framework:** Custom (Nixtla-compatible API)
- **License:** MIT

### Sub-models

| Model | Type | Use Case |
|-------|------|----------|
| Naive MA-30 | Statistical | Baseline reference |
| SARIMAX(1,1,1)(1,1,1,7) | Statistical | Seasonal/intermittent demand |
| XGBoost | Tree-based | Feature interactions |
| LightGBM | Tree-based | Best overall MAPE, fast inference |
| Chronos-2 ZS | Foundation model | Cold-start, zero-shot baseline |
| Routing Ensemble | Meta-model | Combines above via routing logic |

## Intended Use

- **Primary:** SKU-level daily demand forecasting for retail inventory management
- **Users:** Supply chain analysts, inventory planners, ML engineers
- **Scope:** 200 SKUs across 3 warehouses, 4 categories, daily granularity

## Performance

### Benchmark Table

| Model | MAPE | 95% CI | vs Baseline | p-value | Cohen's d | Best For |
|-------|------|--------|-------------|---------|-----------|----------|
| Naive MA-30 | 22.3% | [21.1, 23.5] | — | — | — | Reference |
| SARIMAX | 18.1% | [17.2, 19.0] | −4.2% | 0.002** | 1.2 (L) | Seasonal |
| XGBoost | 14.2% | [13.5, 14.9] | −8.1% | <0.001*** | 2.1 (L) | Features |
| LightGBM | 12.1% | [11.3, 12.9] | −10.2% | <0.001*** | 2.5 (L) | Best single |
| Chronos-2 ZS | 16.4% | [15.8, 17.0] | −5.9% | <0.001*** | 1.5 (L) | Cold-start |
| **Routing Ens.** | **10.3%** | **[9.8, 10.8]** | **−12.0%** | <0.001*** | 3.0 (L) | **Overall** |

### Evaluation Protocol

- 12-fold walk-forward CV (monthly retrain, 14-day horizon)
- Statistical test: Wilcoxon signed-rank vs Naive baseline, α=0.05
- Effect size: Cohen's d (S<0.5, M=0.5-0.8, L>0.8)
- Conformal intervals: 90% target coverage, 91.2% actual

## Limitations

### Known Weaknesses

1. **Synthetic data only:** Model is trained and evaluated on synthetic data with M5-style properties. Real-world performance may differ.
2. **Single-node deployment:** Not tested on distributed systems or high-concurrency scenarios.
3. **Chronos-2 dependency:** Zero-shot baseline requires model download (~300MB). Offline environments need pre-cached model.
4. **SARIMAX scalability:** Fitting SARIMAX per-SKU is slow for >1000 SKUs. Consider batched AutoARIMA.
5. **Feature lag:** Online feature store has eventual consistency (up to 1 day stale). Not suitable for intra-day forecasting.

### Failure Modes

- Cold-start SKUs with <7 days history: only Chronos-2 ZS available, MAPE >16%
- Censored demand (stockouts): model sees 0 demand, not true demand
- Distribution shift: if demand pattern changes, model needs retraining (monitored by Evidently)

### Ethical Considerations

- No PII in training data (synthetic)
- No fairness concerns (forecasting physical goods, not human decisions)
- Potential for inventory bias: over-forecasting leads to waste; under-forecasting leads to stockouts

## Training Data

- **Source:** Synthetic generator with M5-style statistical properties
- **Size:** ~146K rows (200 SKUs × 730 days)
- **Properties:** Intermittent demand (30%), negative binomial distribution, price elasticity, substitution effects, censored demand
- **Seed:** 42 (fully reproducible)

## Caveats and Recommendations

- Always validate MAPE on hold-out data before deployment
- Monitor drift daily using Evidently test suite
- Retrain monthly or when concept drift is detected
- For production: replace SQLite with PostgreSQL, add Redis for feature serving
