# ADR-002: Routing Ensemble Over Stacking/Blending

## Status
Accepted

## Date
2026-03-02

## Context
ChainInsight forecasts demand for 200 SKUs with heterogeneous characteristics (mature, intermittent, cold-start). We need an ensemble strategy to combine multiple forecasting models.

## Decision
We use a **Routing Ensemble** that assigns each SKU to its best-suited model, rather than stacking or blending all model predictions.

Routing logic:
- `history < 60 days` → Chronos-2 zero-shot (no training data needed)
- `intermittency > 50%` → SARIMAX (handles sparse demand)
- `otherwise` → LightGBM (lowest MAPE on stable SKUs)

## Rationale
1. **Interpretability**: "SKU_0042 uses SARIMAX because it has 63% zero-demand days" is much easier to explain to a supply chain manager than "the meta-learner assigned weight 0.37 to SARIMAX".
2. **Computational efficiency**: Only fits relevant models on relevant data subsets, not all models on all data.
3. **Cold-start handling**: Stacking requires all models to produce predictions, but SARIMAX/LightGBM cannot forecast without history. Routing naturally handles this.
4. **Robustness**: Routing threshold sensitivity analysis shows < 0.3% MAPE variation across 50-70 day thresholds — the system is stable.

## Alternatives Considered

### Stacking (meta-learner)
- **Rejected because:** Requires all base models to produce predictions for all SKUs, which is impossible for cold-start SKUs. Also adds a meta-learner that's hard to interpret.

### Simple averaging
- **Rejected because:** Averages good and bad predictions equally. LightGBM at 12.1% MAPE averaged with Naive at 22.3% yields ~17%, worse than LightGBM alone.

### Weighted averaging (learned weights)
- **Partially considered:** Could work but doesn't solve cold-start problem and is less interpretable than explicit routing.

## Consequences
- MAPE drops from 12.1% (best single model) to 10.3% (routing ensemble)
- Each prediction has a clear audit trail: "used X model because Y condition"
- Adding new models requires updating routing logic (not retraining meta-learner)
- Threshold must be validated when data distribution changes significantly
