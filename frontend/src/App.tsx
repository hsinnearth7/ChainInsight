import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { useEffect } from 'react';
import Layout from './components/Layout';
import DashboardPage from './pages/DashboardPage';
import UploadPage from './pages/UploadPage';
import StatsPage from './pages/StatsPage';
import SupplyChainPage from './pages/SupplyChainPage';
import MLPage from './pages/MLPage';
import CapacityPage from './pages/CapacityPage';
import SensingPage from './pages/SensingPage';
import SOPPage from './pages/SOPPage';
import HistoryPage from './pages/HistoryPage';
import { useWebSocket } from './hooks/useWebSocket';
import { useAppStore } from './stores/appStore';
import type { WSMessage } from './types/websocket';

export default function App() {
  const { addWatchdogEvent, darkMode } = useAppStore();

  // Apply dark mode class on mount
  useEffect(() => {
    if (typeof document === 'undefined') {
      return;
    }
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  // Global WS listener for watchdog events
  useWebSocket({
    url: '/ws/global',
    onMessage: (msg: WSMessage) => {
      if (msg.type === 'watchdog:detected') {
        addWatchdogEvent({
          batch_id: msg.batch_id,
          file: (msg.payload.data as Record<string, string>).file || 'unknown',
          timestamp: msg.timestamp,
        });
      }
    },
  });

  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<DashboardPage />} />
          <Route path="upload" element={<UploadPage />} />
          <Route path="stats" element={<StatsPage />} />
          <Route path="supply-chain" element={<SupplyChainPage />} />
          <Route path="ml" element={<MLPage />} />
          <Route path="capacity" element={<CapacityPage />} />
          <Route path="sensing" element={<SensingPage />} />
          <Route path="sop" element={<SOPPage />} />
          <Route path="history" element={<HistoryPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
