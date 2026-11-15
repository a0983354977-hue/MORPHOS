import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import HistoryTimeline from "./components/HistoryTimeline";
import SignalChart from "./components/SignalChart";
import TelemetryStrip from "./components/TelemetryStrip";
import VirtualLog from "./components/VirtualLog";
import type { MorphosEvent, Telemetry, WalRecord } from "./types";

const PAGE_SIZE = 500;

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return response.json() as Promise<T>;
}

async function sendControl(payload: Record<string, unknown>): Promise<unknown> {
  const response = await fetch("/api/control", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.detail ?? "Control request failed");
  return result;
}

export default function App() {
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null);
  const [records, setRecords] = useState<WalRecord[]>([]);
  const [totalRecords, setTotalRecords] = useState(0);
  const [events, setEvents] = useState<MorphosEvent[]>([]);
  const [totalEvents, setTotalEvents] = useState(0);
  const [streamOpen, setStreamOpen] = useState(false);
  const [selectedStep, setSelectedStep] = useState<number | null>(null);
  const [branchMode, setBranchMode] = useState("gain");
  const [busy, setBusy] = useState(false);
  const [notice, setNotice] = useState("");
  const selectedStepRef = useRef(selectedStep);

  useEffect(() => {
    selectedStepRef.current = selectedStep;
  }, [selectedStep]);

  const loadHistoryPage = useCallback(async (step?: number) => {
    let offset = step == null
      ? 0
      : Math.floor(Math.max(0, step - 1) / PAGE_SIZE) * PAGE_SIZE;
    let page = await getJson<{
      total: number;
      records: WalRecord[];
    }>(`/api/history?offset=${offset}&limit=${PAGE_SIZE}`);
    if (step == null && page.total > PAGE_SIZE) {
      offset = page.total - PAGE_SIZE;
      page = await getJson(`/api/history?offset=${offset}&limit=${PAGE_SIZE}`);
    }
    setTotalRecords(page.total);
    setRecords(page.records);
  }, []);

  const loadEvents = useCallback(async () => {
    const page = await getJson<{ total: number; events: MorphosEvent[] }>(
      "/api/events?offset=0&limit=200");
    setTotalEvents(page.total);
    setEvents((current) => {
      const merged = new Map(current.map((item) => [item.id, item]));
      page.events.forEach((item) => merged.set(item.id, item));
      return [...merged.values()].sort((left, right) => left.id - right.id);
    });
  }, []);

  const loadOlderEvents = useCallback(async () => {
    if (events.length >= totalEvents) return 0;
    try {
      const page = await getJson<{ total: number; events: MorphosEvent[] }>(
        `/api/events?offset=${events.length}&limit=200`);
      setTotalEvents(page.total);
      setEvents((current) => {
        const merged = new Map(current.map((item) => [item.id, item]));
        page.events.forEach((item) => merged.set(item.id, item));
        return [...merged.values()].sort((left, right) => left.id - right.id);
      });
      return page.events.length;
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Older SQLite events unavailable");
      return 0;
    }
  }, [events.length, totalEvents]);

  useEffect(() => {
    let active = true;
    let socket: WebSocket | null = null;
    let reconnect: number | undefined;
    const connect = () => {
      if (!active) return;
      const scheme = location.protocol === "https:" ? "wss:" : "ws:";
      socket = new WebSocket(`${scheme}//${location.host}/ws`);
      socket.onopen = () => setStreamOpen(true);
      socket.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data) as Telemetry;
          if (message.type !== "telemetry") return;
          setTelemetry(message);
          const latest = message.core?.latest;
          if (!latest) return;
          setRecords((current) => {
            if (selectedStepRef.current !== null) return current;
            const lastStep = current[current.length - 1]?.step ?? 0;
            if (latest.step <= lastStep) return current;
            const row: WalRecord = {
              step: latest.step,
              timestamp: latest.timestamp,
              confidence: latest.confidence,
              actual: latest.actual,
              predicted: latest.predicted,
              residual: latest.residual,
              rate: latest.rate,
              flags: latest.flags,
              wall: message.observed_at ?? Date.now() / 1000,
            };
            const merged = [...current, row];
            return merged.slice(-5000);
          });
        } catch (error) {
          setNotice(error instanceof Error ? error.message : "Invalid bridge message");
        }
      };
      socket.onclose = () => {
        setStreamOpen(false);
        if (active) reconnect = window.setTimeout(connect, 1000);
      };
      socket.onerror = () => socket?.close();
    };
    connect();
    void loadHistoryPage().catch((error: unknown) =>
      setNotice(error instanceof Error ? error.message : "WAL history unavailable"));
    void loadEvents().catch((error: unknown) =>
      setNotice(error instanceof Error ? error.message : "SQLite events unavailable"));
    const eventTimer = window.setInterval(() => {
      void loadEvents().catch((error: unknown) =>
        setNotice(error instanceof Error ? error.message : "SQLite events unavailable"));
    }, 2000);
    return () => {
      active = false;
      if (reconnect !== undefined) window.clearTimeout(reconnect);
      window.clearInterval(eventTimer);
      socket?.close();
    };
  }, [loadEvents, loadHistoryPage]);

  const selectedRecord = useMemo(
    () => selectedStep === null
      ? records[records.length - 1] ?? null
      : records.find((record) => record.step === selectedStep) ?? null,
    [records, selectedStep],
  );
  const connected = streamOpen && telemetry?.connected === true;
  const core = telemetry?.core;
  const latest = core?.latest;

  const control = async (payload: Record<string, unknown>) => {
    setBusy(true);
    setNotice("");
    try {
      const result = await sendControl(payload);
      setNotice(JSON.stringify(result));
      await loadEvents();
    } catch (error) {
      setNotice(error instanceof Error ? error.message : "Control failed");
    } finally {
      setBusy(false);
    }
  };

  const toggleSource = (source: "lorenz" | "system", enabled: boolean) => {
    void control({ action: "source", source, enabled });
  };

  const selectStep = (step: number) => {
    setSelectedStep(step);
    if (!records.some((record) => record.step === step)) {
      void loadHistoryPage(step).catch((error: unknown) =>
        setNotice(error instanceof Error ? error.message : "WAL page unavailable"));
    }
  };

  return (
    <main className="studio">
      <header className="command-bar">
        <div className="brand">
          <div className={`connection-lamp ${connected ? "is-live" : "is-down"}`} />
          <div>
            <strong>MORPHOS <span>STUDIO</span></strong>
            <small>CORE TELEMETRY / PHASE V</small>
          </div>
        </div>
        <div className="top-metrics">
          <div><span>IPC</span><strong>{connected ? "CONNECTED" : "DISCONNECTED"}</strong></div>
          <div><span>ROUND TRIP</span><strong>{telemetry?.ipc_latency_ms == null ? "—" : `${telemetry.ipc_latency_ms.toFixed(2)} ms`}</strong></div>
          <div><span>NENGO RATE</span><strong>{latest ? `${latest.rate.toFixed(1)} Hz` : "—"}</strong></div>
          <div><span>WAL</span><strong>{core?.wal_records?.toLocaleString() ?? "—"} <i>{connected ? "WRITING" : "—"}</i></strong></div>
        </div>
        <div className="mode-controls">
          <button className={core?.mode === "live" ? "selected" : ""} disabled={!connected || busy}
            onClick={() => void control({ action: "set_mode", mode: "live" })}>LIVE</button>
          <button className={core?.mode === "combat5" ? "danger selected" : "danger"}
            disabled={!connected || busy}
            onClick={() => void control({ action: "set_mode", mode: "combat5" })}>COMBAT5 INJECT</button>
          <button disabled={!connected || busy}
            onClick={() => void control({ action: "reset_breaker" })}>RESET BREAKER</button>
        </div>
      </header>

      <div className="source-bar panel">
        <span className="eyebrow">DATA SOURCES</span>
        <label><input type="checkbox" checked={telemetry?.sources.lorenz ?? false}
          onChange={(event) => toggleSource("lorenz", event.target.checked)} />
          Lorenz-63 <small>deterministic adapter → IPC</small></label>
        <label><input type="checkbox" checked={telemetry?.sources.system ?? false}
          onChange={(event) => toggleSource("system", event.target.checked)} />
          Host resources <small>local psutil telemetry</small></label>
        <span className={`breaker-pill breaker-${(core?.breaker_state ?? "offline").toLowerCase()}`}>
          BREAKER · {core?.breaker_state ?? "OFFLINE"}
        </span>
        <span className="mode-pill">{core?.mode ?? "NO CORE"}</span>
      </div>

      {notice && <div className="notice" role="status">{notice}</div>}

      <TelemetryStrip telemetry={telemetry} />

      <div className="workspace-grid">
        <section className="main-column">
          <section className="panel chart-panel">
            <header className="section-heading">
              <div><span className="eyebrow">ACTUAL WAL RECORDS · Z-SCORED STATE</span><h2>Input vs. one-step prediction</h2></div>
              <span className="chart-legend"><i className="legend-actual" />actual <i className="legend-predicted" />predicted</span>
            </header>
            <SignalChart records={records} metric="state" />
          </section>

          <div className="lower-grid">
            <section className="panel chart-panel">
              <header className="section-heading">
                <div><span className="eyebrow">NEURAL READOUT</span><h2>Prediction residual</h2></div>
                <strong className={latest?.anomaly ? "value-danger" : "value-good"}>
                  {latest ? latest.residual.toPrecision(4) : "—"}
                </strong>
              </header>
              <SignalChart records={records} metric="residual" />
              <div className="anomaly-band">
                <span>Dual-Opinion</span>
                <strong>{latest?.anomaly_score == null ? "not calibrated" : latest.anomaly_score.toFixed(3)}</strong>
                <span>threshold</span>
                <strong>{latest?.anomaly_threshold == null ? "—" : latest.anomaly_threshold.toFixed(3)}</strong>
                <i className={latest?.anomaly ? "alarm-active" : ""} />
              </div>
            </section>

            <section className="panel geometry-panel">
              <header className="section-heading">
                <div><span className="eyebrow">DSW CROSS-TALK MONITOR</span><h2>Orthogonal subspaces</h2></div>
                <strong className="value-good">
                  {telemetry?.geometry ? telemetry.geometry.basis_cross_talk.toExponential(2) : "—"}
                </strong>
              </header>
              <div className="subspace-readout">
                <div><span>Lorenz energy</span><strong>{telemetry?.geometry?.lorenz_energy?.toFixed(4) ?? "—"}</strong></div>
                <div><span>System energy</span><strong>{telemetry?.geometry?.system_energy?.toFixed(4) ?? "—"}</strong></div>
              </div>
              <p>Basis overlap measured from the configured orthogonal projector. Source energies are shown only when their real streams are available.</p>
            </section>
          </div>

          <HistoryTimeline records={records} total={Math.max(totalRecords, core?.wal_records ?? 0)}
            selectedStep={selectedStep} onSelect={selectStep} />
        </section>

        <aside className="side-column">
          <section className="panel selected-panel">
            <header className="section-heading">
              <div><span className="eyebrow">PERSISTED SAMPLE</span><h2>{selectedRecord ? `STEP ${selectedRecord.step}` : "No WAL sample"}</h2></div>
              <div className="sample-actions">
              <button disabled={selectedStep === null} onClick={() => {
                setSelectedStep(null);
                void loadHistoryPage().catch((error: unknown) =>
                  setNotice(error instanceof Error ? error.message : "Latest WAL page unavailable"));
              }}>RETURN LIVE</button>
              <button className="branch-button" disabled={!connected || !selectedRecord || busy}
                onClick={() => void control({
                  action: "branch", branch_mode: branchMode,
                  step: selectedRecord?.step,
                })}>BRANCH SNAPSHOT</button>
              </div>
            </header>
            <div className="branch-controls">
              <label htmlFor="branch-mode">Counterfactual perturbation</label>
              <select id="branch-mode" value={branchMode} onChange={(event) => setBranchMode(event.target.value)}>
                <option value="gain">Input gain ×1.6</option>
                <option value="noise">Seeded input noise</option>
                <option value="step">Input offset +0.8</option>
                <option value="dropout">Input dropout</option>
              </select>
            </div>
            <div className="sample-grid">
              <span>time</span><strong>{selectedRecord ? `${selectedRecord.timestamp.toFixed(3)} s` : "—"}</strong>
              <span>confidence</span><strong>{selectedRecord?.confidence.toFixed(3) ?? "—"}</strong>
              <span>firing rate</span><strong>{selectedRecord ? `${selectedRecord.rate.toFixed(1)} Hz` : "—"}</strong>
              <span>flags</span><strong>0x{selectedRecord?.flags.toString(16).padStart(2, "0") ?? "—"}</strong>
              <span>actual z</span><strong>{selectedRecord?.actual.map((n) => n.toFixed(3)).join(" / ") ?? "—"}</strong>
              <span>predicted z</span><strong>{selectedRecord?.predicted.map((n) => n.toFixed(3)).join(" / ") ?? "—"}</strong>
            </div>
            {core?.branches.map((branch) => (
              <div className="branch-result" key={branch.name}>
                <b>{branch.name}</b><span>{branch.steps} branch steps</span>
                <span>residual {branch.latest_residual?.toPrecision(4) ?? "—"}</span>
              </div>
            ))}
          </section>
          <VirtualLog events={events} total={totalEvents} onLoadOlder={loadOlderEvents} />
          <div className="bridge-footnote">
            <span>20 Hz bridge push</span>
            <span>{telemetry?.observed_at ? new Date(telemetry.observed_at * 1000).toLocaleTimeString() : "waiting for daemon"}</span>
          </div>
        </aside>
      </div>
    </main>
  );
}
