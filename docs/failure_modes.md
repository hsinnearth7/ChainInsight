# Failure Mode Analysis

## System Components and Their Failure Modes

### 1. Data Pipeline

| Component | Failure Mode | Detection | Impact | Mitigation |
|-----------|-------------|-----------|--------|------------|
| CSV ingestion | Malformed CSV | Pandera schema validation | Pipeline halts for batch | Return 400 with validation errors |
| Data generator | Seed mismatch | SHA-256 hash check | Non-reproducible results | Enforce PYTHONHASHSEED=42 |
| ETL pipeline | OOM on large file | Memory monitoring | Batch fails | Chunk processing, max file size |
| Feature computation | NaN propagation | Pandera nullable=False | Corrupt features | fillna + validation checkpoint |

### 2. Forecasting Models

| Component | Failure Mode | Detection | Impact | Mitigation |
|-----------|-------------|-----------|--------|------------|
| SARIMAX | Non-convergence | maxiter check + try/except | No forecast for SKU | Fallback to Naive MA-30 |
| LightGBM | Insufficient training data | Min samples check | Poor predictions | Route to Chronos-2 ZS |
| Chronos-2 | Model download failure | ImportError catch | No ZS baseline | Fallback to Naive MA-30 |
| Routing Ensemble | All sub-models fail | Empty prediction check | No ensemble output | Serve Naive baseline + alert |
| Hierarchical reconciliation | Singular S matrix | LinAlgError catch | Inconsistent hierarchy | Fallback to BottomUp |

### 3. RL Training

| Component | Failure Mode | Detection | Impact | Mitigation |
|-----------|-------------|-----------|--------|------------|
| PPO training | Divergence | Reward trend monitoring | Useless policy | Early stopping + checkpoint |
| Curriculum transfer | Env mismatch | Action space validation | Transfer fails | Reinitialize from scratch |
| Stockpyl import | Package not installed | ImportError | No theoretical baseline | Manual newsvendor calculation |

### 4. MLOps / Monitoring

| Component | Failure Mode | Detection | Impact | Mitigation |
|-----------|-------------|-----------|--------|------------|
| Drift detection | False positive | Threshold calibration | Unnecessary retrain | Require 7 consecutive days |
| Auto retrain | Retrain loop | Max retrain counter | Wasted compute | Max 1 retrain per week |
| Feature Store offline | Stale features | TTL check on online store | Slightly stale predictions | Serve stale (< 0.1% impact) |

### 5. Infrastructure

| Component | Failure Mode | Detection | Impact | Mitigation |
|-----------|-------------|-----------|--------|------------|
| SQLite | DB locked | Timeout exception | Writes fail | Retry with backoff, WAL mode |
| WebSocket | Connection drop | Heartbeat timeout | No real-time updates | Auto-reconnect in frontend |
| Docker | Container OOM | Docker healthcheck | Service down | Memory limits + restart policy |
| API rate limiter | Memory growth | Periodic cleanup | Slow memory leak | Prune old entries hourly |

## Degradation Levels

| Level | Condition | Behavior |
|-------|-----------|----------|
| **L0: Normal** | All systems healthy | Full functionality |
| **L1: Partial** | 1 warehouse pipeline fails | Stale forecasts for failed warehouse |
| **L2: Degraded** | Feature store offline | Serve with cached features |
| **L3: Minimal** | All models fail | Serve Naive baseline + urgent alert |
| **L4: Unavailable** | Database corruption | Return 503, trigger recovery |
