# ADR-001: Feature Store Consistency Model (AP > CP)

## Status
Accepted

## Date
2026-03-02

## Context
ChainInsight's Feature Store serves features to both batch training (offline) and real-time API (online). We need to decide the consistency model in the context of the CAP theorem.

The Feature Store has two stores:
- **Offline store**: batch ETL runs daily, produces features for model training
- **Online store**: serves pre-computed features for real-time prediction API

The question: when the offline store updates, how quickly must the online store reflect those changes?

## Decision
We choose **AP (Availability + Partition tolerance) over CP (Consistency + Partition tolerance)**.

Specifically: **Eventual Consistency** with up to 1-day lag between offline and online stores.

## Rationale

### Why availability > consistency for forecasting:
1. **Forecasting is tolerant to stale features.** A 1-day lag in features like `rolling_mean_7` changes predictions by < 0.1% MAPE. This is within noise.
2. **Availability matters more.** If the API returns 503 because the feature store is synchronizing, the downstream inventory system has no forecast at all — much worse than a slightly stale one.
3. **The alternative (CP) requires distributed locks.** For a single-node SQLite system, this adds unnecessary complexity. Even for future PostgreSQL migration, strong consistency requires 2PC or consensus protocols that add latency.

### Why this won't cause problems:
- Feature TTL = 1 day, matching the offline ETL frequency
- Stale features only affect non-critical lag features (lag_1 may be 1 day old)
- Calendar features (day_of_week, month) are deterministic and never stale
- Model retraining always uses offline store (always consistent with itself)

## Alternatives Considered

### CP (Strong Consistency)
- **Rejected because:** Requires distributed locking or synchronous writes. For a demo project with single-node SQLite, this is unnecessary complexity. The consistency guarantee provides < 0.1% MAPE improvement — not worth the availability risk.

### No Feature Store (compute on-the-fly)
- **Rejected because:** Introduces training-serving skew. Without a shared feature computation layer, the training pipeline and API may compute features differently (e.g., different lag window implementations), leading to hard-to-debug prediction errors.

## Consequences
- Online predictions may use features up to 1 day old
- System remains available even during offline ETL batch runs
- No distributed locking complexity
- Clear documentation of the trade-off for interview discussion
