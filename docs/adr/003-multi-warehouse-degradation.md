# ADR-003: Multi-Warehouse Graceful Degradation Strategy

## Status
Accepted

## Date
2026-03-02

## Context
ChainInsight runs independent forecast pipelines per warehouse (NYC, LAX, CHI), then reconciles at the national level via hierarchical forecasting. We need a strategy for when one or more warehouse pipelines fail.

## Decision
Implement **graceful degradation** with automatic fallback:

1. Each warehouse runs an independent forecast pipeline
2. National reconciliation waits for all warehouses (with timeout)
3. If a warehouse fails:
   - Other warehouses continue independently
   - Failed warehouse uses **previous round's forecast** (stale but available)
   - Alert sent to monitoring dashboard
   - Reconciliation proceeds with mix of fresh + stale forecasts

## Rationale
1. **Availability over consistency**: A stale forecast is better than no forecast. Supply chain decisions cannot wait for a failed pipeline to recover.
2. **Blast radius isolation**: NYC pipeline failure should not block LAX inventory decisions.
3. **SRE best practice**: Matches Google SRE's "graceful degradation" pattern — reduce quality, not availability.

## Failure Modes Analyzed

| Failure | Impact | Mitigation |
|---------|--------|------------|
| Single warehouse ETL fails | 33% of forecasts stale | Use previous forecast + alert |
| All warehouses fail | No fresh forecasts | Serve cached forecasts, urgent alert |
| Reconciliation fails | Inconsistent hierarchy | Skip reconciliation, serve per-warehouse |
| Feature store offline | All features stale | Serve with stale features (< 0.1% impact) |

## Alternatives Considered

### Fail-fast (abort all if one fails)
- **Rejected because:** 2 healthy warehouses would be blocked by 1 failure. Unacceptable for operations.

### Retry with backoff
- **Partially adopted:** We retry failed pipelines 3 times with exponential backoff. But after 3 failures, we fall back to stale forecasts rather than blocking.

## Consequences
- System stays available even during partial failures
- Monitoring must track "stale forecast" events
- Reconciliation quality degrades with stale data (quantified: < 2% MAPE impact from 1-day-old forecasts)
