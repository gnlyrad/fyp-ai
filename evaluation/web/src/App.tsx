import { Navigate, Route, Routes } from "react-router-dom";
import { ErrorPanel, Loading } from "./components/ui";
import { useMeta } from "./lib/api";
import { MetaProvider } from "./lib/meta";
import ComparePage from "./pages/ComparePage";
import DatasetPage from "./pages/DatasetPage";
import DatasetsPage from "./pages/DatasetsPage";
import ResultPage from "./pages/ResultPage";
import RunPage from "./pages/RunPage";
import SttCallPage from "./pages/stt/SttCallPage";
import SttDatasetPage from "./pages/stt/SttDatasetPage";
import SttDatasetsPage from "./pages/stt/SttDatasetsPage";
import SttRunPage from "./pages/stt/SttRunPage";

export default function App() {
  const meta = useMeta();
  if (meta.isPending) return <Loading label="Loading dashboard" />;
  if (meta.isError) return <div className="p-6"><ErrorPanel error={meta.error} /></div>;
  return (
    <MetaProvider meta={meta.data}>
      <Routes>
        <Route path="/" element={<Navigate to={`/${meta.data.default_source}`} replace />} />
        <Route path="/stt" element={<SttDatasetsPage />} />
        <Route path="/stt/d/:datasetId" element={<SttDatasetPage />} />
        <Route path="/stt/d/:datasetId/r/:runId" element={<SttRunPage />} />
        <Route path="/stt/d/:datasetId/r/:runId/c/:callId" element={<SttCallPage />} />
        <Route path="/:source" element={<DatasetsPage />} />
        <Route path="/:source/d/:datasetId" element={<DatasetPage />} />
        <Route path="/:source/d/:datasetId/compare" element={<ComparePage />} />
        <Route path="/:source/d/:datasetId/r/:runId" element={<RunPage />} />
        <Route path="/:source/d/:datasetId/r/:runId/result/:callId/:questionId" element={<ResultPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </MetaProvider>
  );
}
