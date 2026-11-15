import type { Telemetry } from "../types";

interface Props {
  telemetry: Telemetry | null;
}

function metric(value: number | undefined, digits = 2): string {
  return value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
}

export default function TelemetryStrip({ telemetry }: Props) {
  const system = telemetry?.system;
  const latest = telemetry?.core?.latest;
  const cards = [
    { label: "CPU", value: system ? `${metric(system[0] * 100, 1)}%` : "—", source: "psutil" },
    { label: "MEMORY", value: system ? `${metric(system[1] * 100, 1)}%` : "—", source: "psutil" },
    { label: "DISK I/O", value: system ? metric(system[2], 3) : "—", source: "log rate" },
    { label: "NETWORK I/O", value: system ? metric(system[3], 3) : "—", source: "log rate" },
    { label: "NEURAL RATE", value: latest ? `${metric(latest.rate, 1)} Hz` : "—", source: "Nengo WAL" },
    { label: "CONFIDENCE", value: latest ? metric(latest.confidence, 3) : "—", source: "TimeAligner" },
    { label: "BASIS LEAK", value: telemetry?.geometry ? metric(telemetry.geometry.basis_cross_talk, 6) : "—", source: "orthogonal projector" },
  ];

  return (
    <section className="telemetry-strip" aria-label="Telemetry metrics">
      {cards.map((card) => (
        <article className="metric-card" key={card.label}>
          <span className="metric-label">{card.label}</span>
          <strong>{card.value}</strong>
          <span className="metric-source">{card.source}</span>
        </article>
      ))}
    </section>
  );
}
