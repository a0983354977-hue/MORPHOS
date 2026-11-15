import type { WalRecord } from "../types";

interface Props {
  records: WalRecord[];
  total: number;
  selectedStep: number | null;
  onSelect: (step: number) => void;
}

export default function HistoryTimeline({
  records, total, selectedStep, onSelect,
}: Props) {
  const first = records[0]?.step ?? 0;
  const last = records[records.length - 1]?.step ?? 0;
  const value = selectedStep ?? last;
  return (
    <section className="panel timeline-panel">
      <header className="section-heading">
        <div>
          <span className="eyebrow">BINARY WAL • PAGED READ</span>
          <h2>History timeline</h2>
        </div>
        <span className="timeline-count">{total.toLocaleString()} records</span>
      </header>
      <div className="timeline-scroll">
        <div className="timeline-range">
          <span>step {first.toLocaleString()}</span>
          <input
            type="range"
            min={1}
            max={Math.max(total, 1)}
            value={Math.min(value, Math.max(total, 1))}
            onChange={(event) => onSelect(Number(event.target.value))}
            aria-label="Scrub WAL step"
          />
          <span>step {last.toLocaleString()}</span>
        </div>
        <div className="timeline-bars" aria-label="WAL sample residual markers">
          {records.slice(-180).map((record) => (
            <button
              className={`timeline-bar ${record.flags & 128 ? "bar-breaker" : ""} ${record.flags & 256 ? "bar-anomaly" : ""}`}
              key={record.step}
              style={{ height: `${Math.min(100, 8 + Math.sqrt(Math.abs(record.residual)) * 18)}%` }}
              title={`step ${record.step} · residual ${record.residual.toPrecision(4)}`}
              onClick={() => onSelect(record.step)}
            />
          ))}
        </div>
      </div>
      <footer className="panel-foot">
        Scroll the strip horizontally; selecting a step loads its actual persisted record.
      </footer>
    </section>
  );
}
