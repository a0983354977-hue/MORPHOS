import { useEffect, useRef, useState } from "react";
import type { MorphosEvent } from "../types";

const ROW_HEIGHT = 32;

interface Props {
  events: MorphosEvent[];
  total: number;
  onLoadOlder: () => Promise<number>;
}

export default function VirtualLog({ events, total, onLoadOlder }: Props) {
  const viewport = useRef<HTMLDivElement>(null);
  const [scrollTop, setScrollTop] = useState(0);
  const [followLatest, setFollowLatest] = useState(true);
  const [loadingOlder, setLoadingOlder] = useState(false);
  const visibleCount = Math.ceil(288 / ROW_HEIGHT) + 2;
  const start = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - 1);
  const end = Math.min(events.length, start + visibleCount);

  useEffect(() => {
    if (followLatest && viewport.current) {
      viewport.current.scrollTop = viewport.current.scrollHeight;
    }
  }, [events, followLatest]);

  return (
    <section className="panel log-panel">
      <header className="section-heading">
        <div>
          <span className="eyebrow">PERSISTED EVENTS</span>
          <h2>Event & anomaly log</h2>
        </div>
        <label className="follow-toggle">
          <input
            type="checkbox"
            checked={followLatest}
            onChange={(event) => setFollowLatest(event.target.checked)}
          />
          follow latest
        </label>
      </header>
      <div
        className="virtual-viewport"
        ref={viewport}
        onScroll={(event) => {
          const element = event.currentTarget;
          setScrollTop(element.scrollTop);
          setFollowLatest(
            element.scrollHeight - element.scrollTop - element.clientHeight < 16,
          );
          if (element.scrollTop < 24 && events.length < total && !loadingOlder) {
            setLoadingOlder(true);
            void onLoadOlder()
              .then((count) => {
                if (count > 0 && viewport.current) {
                  viewport.current.scrollTop += count * ROW_HEIGHT;
                }
              })
              .finally(() => setLoadingOlder(false));
          }
        }}
      >
        {events.length === 0 ? (
          <div className="empty-row">No persisted events yet</div>
        ) : (
          <>
            <div style={{ height: start * ROW_HEIGHT }} />
            {events.slice(start, end).map((item) => (
              <div className="event-row" key={item.id} style={{ height: ROW_HEIGHT }}>
                <time>{item.sim_t == null ? "—" : `${item.sim_t.toFixed(3)}s`}</time>
                <span className={`event-level level-${item.level.toLowerCase()}`}>
                  {item.level}
                </span>
                <span className="event-message" title={item.message}>
                  {item.message}
                </span>
                <span className="event-data">
                  {typeof item.data?.score === "number"
                    ? `score ${item.data.score.toFixed(3)}`
                    : `#${item.id}`}
                </span>
              </div>
            ))}
            <div style={{ height: Math.max(0, events.length - end) * ROW_HEIGHT }} />
          </>
        )}
      </div>
      <footer className="panel-foot">
        {events.length.toLocaleString()} / {total.toLocaleString()} loaded from SQLite
        {loadingOlder ? " · loading older records" : ""}
      </footer>
    </section>
  );
}
