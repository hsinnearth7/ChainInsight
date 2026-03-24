import { useEffect, useState } from 'react';
import { api } from '../api/client';
import { useAppStore } from '../stores/appStore';
import KPICard from '../components/KPICard';
import ChartImage from '../components/ChartImage';
import DataTable from '../components/DataTable';
import LoadingSpinner from '../components/LoadingSpinner';
import { useTranslation } from '../i18n/useTranslation';
import type { AnalysisResult } from '../types/api';

export default function SOPPage() {
  const [batchId, setBatchId] = useState<string | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(true);
  const latestBatchId = useAppStore((s) => s.latestBatchId);
  const { t } = useTranslation();

  useEffect(() => {
    loadData();
  }, [latestBatchId]);

  async function loadData() {
    setLoading(true);
    try {
      let bid = latestBatchId;
      if (!bid) {
        const kpis = await api.getLatestKPIs();
        bid = kpis.batch_id;
      }
      setBatchId(bid);
      const data = await api.getAnalysis(bid, 'sop');
      setAnalysis(data);
    } catch {
      // no data
    } finally {
      setLoading(false);
    }
  }

  if (loading) return <LoadingSpinner text={t('sop.loading')} />;
  if (!analysis) return <div className="text-ci-gray text-center py-12">{t('sop.noData')}</div>;

  const sopData = (analysis.kpis || {}) as Record<string, unknown>;
  const scenarios = (sopData.scenarios || []) as Record<string, unknown>[];

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">{t('sop.title')}</h2>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <KPICard
          title={t('sop.fillRate')}
          value={typeof sopData.fill_rate === 'number' ? `${((sopData.fill_rate as number) * 100).toFixed(1)}%` : '--'}
          icon="FIL"
          color="ci-success"
        />
        <KPICard
          title={t('sop.scenarios')}
          value={String(scenarios.length || 3)}
          icon="SCN"
          color="ci-primary"
        />
        <KPICard
          title={t('capacity.utilization')}
          value={typeof sopData.avg_utilization === 'number' ? `${((sopData.avg_utilization as number) * 100).toFixed(1)}%` : '--'}
          icon="UTL"
          color="ci-teal"
        />
        <KPICard
          title={t('sop.demandSupplyBalance')}
          value={typeof sopData.balance_pct === 'number' ? `${((sopData.balance_pct as number) * 100).toFixed(1)}%` : '--'}
          icon="BAL"
          color="ci-purple"
        />
      </div>

      {scenarios.length > 0 && (
        <DataTable
          data={scenarios}
          title={t('sop.comparison')}
          columns={['scenario', 'fill_rate', 'utilization', 'inventory_cost', 'stockout_risk']}
        />
      )}

      {batchId && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ChartImage src={api.getChartURL(batchId, 'chart_sop_balance.png')} alt="Demand-Supply Balance" />
          <ChartImage src={api.getChartURL(batchId, 'chart_sop_scenarios.png')} alt="Scenario Comparison" />
        </div>
      )}
    </div>
  );
}
