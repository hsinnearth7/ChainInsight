import { useEffect, useState } from 'react';
import { api } from '../api/client';
import { useAppStore } from '../stores/appStore';
import { usePipelineProgress } from '../hooks/usePipelineProgress';
import FileUploader from '../components/FileUploader';
import PipelineProgress from '../components/PipelineProgress';
import KPICard from '../components/KPICard';
import { useTranslation } from '../i18n/useTranslation';

export default function UploadPage() {
  const [batchId, setBatchId] = useState<string | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState('');
  const [etlSummary, setEtlSummary] = useState<Record<string, unknown> | null>(null);

  const setLatestBatchId = useAppStore((s) => s.setLatestBatchId);
  const setActivePipelineBatchId = useAppStore((s) => s.setActivePipelineBatchId);
  const { stages, overallPct, pipelineStatus, connected, reset } = usePipelineProgress(batchId);
  const { t } = useTranslation();

  async function handleUpload(file: File) {
    setUploading(true);
    setUploadError('');
    setEtlSummary(null);
    reset();

    try {
      const result = await api.ingest(file);
      setBatchId(result.batch_id);
      setActivePipelineBatchId(result.batch_id);
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : 'Upload failed');
    } finally {
      setUploading(false);
    }
  }

  async function handleUseExisting() {
    setUploading(true);
    setUploadError('');
    setEtlSummary(null);
    reset();

    try {
      const result = await api.ingestExisting();
      setBatchId(result.batch_id);
      setActivePipelineBatchId(result.batch_id);
    } catch (e: unknown) {
      setUploadError(e instanceof Error ? e.message : 'Failed to trigger pipeline');
    } finally {
      setUploading(false);
    }
  }

  useEffect(() => {
    if (pipelineStatus !== 'completed' || etlSummary || !batchId) {
      return;
    }

    api.getAnalysis(batchId, 'etl').then((r) => setEtlSummary(r.kpis)).catch(() => {});
  }, [batchId, etlSummary, pipelineStatus]);

  useEffect(() => {
    if (!batchId) {
      return;
    }

    if (pipelineStatus === 'completed') {
      setLatestBatchId(batchId);
      setActivePipelineBatchId(null);
    } else if (pipelineStatus === 'failed') {
      setActivePipelineBatchId(null);
    }
  }, [batchId, pipelineStatus, setActivePipelineBatchId, setLatestBatchId]);

  return (
    <div className="space-y-6 max-w-2xl mx-auto">
      <h2 className="text-xl font-bold">{t('upload.title')}</h2>

      <FileUploader onFileSelected={handleUpload} disabled={uploading || pipelineStatus === 'running'} />

      <div className="text-center">
        <button
          onClick={handleUseExisting}
          disabled={uploading || pipelineStatus === 'running'}
          className="text-sm text-ci-primary hover:underline disabled:opacity-50"
        >
          {t('upload.useExisting')}
        </button>
      </div>

      {uploadError && (
        <div className="bg-ci-danger/10 text-ci-danger text-sm rounded-lg px-4 py-3">{uploadError}</div>
      )}

      {batchId && (
        <>
          <div className="text-xs text-ci-gray">
            Batch ID: <code className="font-mono">{batchId}</code>
          </div>
          <PipelineProgress
            stages={stages}
            overallPct={overallPct}
            pipelineStatus={pipelineStatus}
            connected={connected}
          />
        </>
      )}

      {pipelineStatus === 'completed' && etlSummary && (
        <div className="space-y-3">
          <h3 className="text-sm font-medium">{t('upload.etlSummary')}</h3>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
            {Object.entries(etlSummary).map(([key, val]) => (
              <KPICard
                key={key}
                title={key.replace(/_/g, ' ')}
                value={typeof val === 'number' ? val.toLocaleString() : String(val ?? '--')}
              />
            ))}
          </div>
        </div>
      )}

      {pipelineStatus === 'completed' && (
        <div className="text-center text-ci-success text-sm font-medium">
          {t('upload.pipelineComplete')}
        </div>
      )}
    </div>
  );
}
