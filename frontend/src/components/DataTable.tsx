import { useState, useMemo } from 'react';

interface DataTableProps {
  data: Record<string, unknown>[];
  columns?: string[];
  title?: string;
  enableCSVDownload?: boolean;
  maxHeight?: string;
}

export default function DataTable({
  data,
  columns,
  title,
  enableCSVDownload = false,
  maxHeight = '400px',
}: DataTableProps) {
  const [sortCol, setSortCol] = useState<string | null>(null);
  const [sortAsc, setSortAsc] = useState(true);

  const cols = columns || (data.length > 0 ? Object.keys(data[0]) : []);

  const sorted = useMemo(() => {
    if (!sortCol) return data;
    return [...data].sort((a, b) => {
      const va = a[sortCol] as string | number;
      const vb = b[sortCol] as string | number;
      if (va === vb) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      const cmp =
        typeof va === 'number' && typeof vb === 'number'
          ? va - vb
          : String(va).localeCompare(String(vb));
      return sortAsc ? cmp : -cmp;
    });
  }, [data, sortCol, sortAsc]);

  function handleSort(col: string) {
    if (sortCol === col) {
      setSortAsc(!sortAsc);
    } else {
      setSortCol(col);
      setSortAsc(true);
    }
  }

  function downloadCSV() {
    const header = cols.join(',');
    const rows = data.map((row) => cols.map((c) => JSON.stringify(row[c] ?? '')).join(','));
    const csv = [header, ...rows].join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${title || 'data'}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }

  if (!data.length) {
    return <div className="text-ci-gray text-sm">No data available</div>;
  }

  return (
    <div className="bg-white dark:bg-ci-dark-card rounded-lg border border-gray-200 dark:border-gray-700">
      {(title || enableCSVDownload) && (
        <div className="flex items-center justify-between px-4 py-2 border-b border-gray-200 dark:border-gray-700">
          {title && <h3 className="text-sm font-medium">{title}</h3>}
          {enableCSVDownload && (
            <button onClick={downloadCSV} className="text-xs text-ci-primary hover:underline">
              Download CSV
            </button>
          )}
        </div>
      )}
      <div className="overflow-auto" style={{ maxHeight }}>
        <table className="w-full text-sm">
          <thead className="bg-gray-50 dark:bg-gray-800 sticky top-0">
            <tr>
              {cols.map((col) => (
                <th
                  key={col}
                  className="px-3 py-2 text-left text-xs font-medium text-ci-gray uppercase tracking-wider cursor-pointer hover:text-ci-primary whitespace-nowrap"
                  onClick={() => handleSort(col)}
                >
                  {col.replace(/_/g, ' ')}
                  {sortCol === col && (sortAsc ? ' ▲' : ' ▼')}
                </th>
              ))}
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-100 dark:divide-gray-700">
            {sorted.map((row, i) => (
              <tr key={i} className="hover:bg-gray-50 dark:hover:bg-gray-700/30">
                {cols.map((col) => (
                  <td key={col} className="px-3 py-1.5 whitespace-nowrap">
                    {formatCell(row[col])}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function formatCell(val: unknown): string {
  if (val == null) return '--';
  if (typeof val === 'number') return val % 1 === 0 ? String(val) : val.toFixed(2);
  return String(val);
}
