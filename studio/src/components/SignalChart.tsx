import { useEffect, useMemo, useRef } from "react";
import uPlot, { type AlignedData } from "uplot";
import "uplot/dist/uPlot.min.css";
import type { WalRecord } from "../types";

interface Props {
  records: WalRecord[];
  metric: "state" | "residual";
}

export default function SignalChart({ records, metric }: Props) {
  const host = useRef<HTMLDivElement>(null);
  const chart = useRef<uPlot | null>(null);
  const data = useMemo<AlignedData>(() => {
    if (metric === "residual") {
      return [
        records.map((row) => row.timestamp),
        records.map((row) => row.residual),
      ];
    }
    return [
      records.map((row) => row.timestamp),
      ...[0, 1, 2].flatMap((axis) => [
        records.map((row) => row.actual[axis] ?? Number.NaN),
        records.map((row) => row.predicted[axis] ?? Number.NaN),
      ]),
    ] as AlignedData;
  }, [records, metric]);

  useEffect(() => {
    if (!host.current) return;
    const stateSeries = [
      { label: "time" },
      { label: "x actual", stroke: "#2dd4bf", width: 1.5 },
      { label: "x predicted", stroke: "#5eead4", width: 1, dash: [4, 3] },
      { label: "y actual", stroke: "#60a5fa", width: 1.5 },
      { label: "y predicted", stroke: "#93c5fd", width: 1, dash: [4, 3] },
      { label: "z actual", stroke: "#fbbf24", width: 1.5 },
      { label: "z predicted", stroke: "#fde68a", width: 1, dash: [4, 3] },
    ];
    const options: uPlot.Options = {
      width: Math.max(host.current.clientWidth, 320),
      height: metric === "state" ? 260 : 170,
      series: metric === "state"
        ? stateSeries
        : [{ label: "time" }, { label: "squared residual", stroke: "#f59e0b", width: 1.5 }],
      axes: [
        { stroke: "#718096", grid: { stroke: "#202a33", width: 1 } },
        { stroke: "#718096", grid: { stroke: "#202a33", width: 1 } },
      ],
      cursor: { drag: { x: true, y: false } },
      select: { left: 0, top: 0, width: 0, height: 0, show: false },
      scales: { x: { time: false } },
    };
    chart.current = new uPlot(options, data, host.current);
    const observer = new ResizeObserver(() => {
      if (host.current && chart.current) {
        chart.current.setSize({
          width: Math.max(host.current.clientWidth, 320),
          height: metric === "state" ? 260 : 170,
        });
      }
    });
    observer.observe(host.current);
    return () => {
      observer.disconnect();
      chart.current?.destroy();
      chart.current = null;
    };
  }, [metric]);

  useEffect(() => {
    chart.current?.setData(data);
  }, [data]);

  return (
    <div className="chart-shell">
      {records.length === 0 && <div className="chart-empty">Waiting for WAL records</div>}
      <div className="chart-host" ref={host} />
    </div>
  );
}
