import { useEffect, useState } from 'react';
import { api } from '../api/client';
import { useAppStore } from '../stores/appStore';
import KPICard from '../components/KPICard';
import ChartImage from '../components/ChartImage';
import LoadingSpinner from '../components/LoadingSpinner';
import { useTranslation } from '../i18n/useTranslation';
import type { AnalysisResult } from '../types/api';

export default function SensingPage() {
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
      const data = await api.getAnalysis(bid, 'sensing');
      setAnalysis(data);
    } catch {
      // no data
    } finally {
      setLoading(false);
    }
  }

  if (loading) return <LoadingSpinner text={t('sensing.loading')} />;
  if (!analysis) return <div className="text-ci-gray text-center py-12">{t('sensing.noData')}</div>;

  const senseData = (analysis.kpis || {}) as Record<string, unknown>;

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">{t('sensing.title')}</h2>

      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <KPICard
          title={t('sensing.signals')}
          value={typeof senseData.active_signals === 'number' ? String(senseData.active_signals) : '0'}
          icon="SIG"
          color="ci-primary"
        />
        <KPICard
          title={t('sensing.adjustment')}
          value={typeof senseData.avg_adjustment_pct === 'number' ? `${(senseData.avg_adjustment_pct as number).toFixed(1)}%` : '--'}
          icon="ADJ"
          color="ci-teal"
        />
        <KPICard
          title={t('sensing.spikes')}
          value={typeof senseData.spike_count === 'number' ? String(senseData.spike_count) : '0'}
          icon="SPK"
          color="ci-warning"
        />
      </div>

      {batchId && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <ChartImage src={api.getChartURL(batchId, 'chart_sensing_signals.png')} alt="Signal Timeline" />
          <ChartImage src={api.getChartURL(batchId, 'chart_sensing_adjustment.png')} alt="Forecast Adjustment" />
        </div>
      )}
    </div>
  );
}
