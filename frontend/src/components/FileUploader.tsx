import { useState, useCallback, useRef } from 'react';
import { useTranslation } from '../i18n/useTranslation';

interface FileUploaderProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export default function FileUploader({ onFileSelected, disabled = false }: FileUploaderProps) {
  const [dragOver, setDragOver] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const { t } = useTranslation();

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      if (disabled) return;
      const file = e.dataTransfer.files[0];
      if (file && file.name.endsWith('.csv')) {
        onFileSelected(file);
      }
    },
    [onFileSelected, disabled]
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      if (file) onFileSelected(file);
    },
    [onFileSelected]
  );

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors cursor-pointer
        ${dragOver ? 'border-ci-primary bg-ci-primary/5' : 'border-gray-300 dark:border-gray-600 hover:border-ci-primary/50'}
        ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current?.click()}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".csv"
        className="hidden"
        onChange={handleChange}
        disabled={disabled}
      />
      <div className="text-4xl mb-3 font-semibold tracking-wide">CSV</div>
      <p className="text-sm font-medium">{t('upload.dragDrop')}</p>
      <p className="text-xs text-ci-gray mt-1">{t('upload.browse')}</p>
    </div>
  );
}
