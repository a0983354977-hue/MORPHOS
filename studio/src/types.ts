export interface WalRecord {
  step: number;
  timestamp: number;
  confidence: number;
  actual: number[];
  predicted: number[];
  residual: number;
  rate: number;
  flags: number;
  wall: number;
}

export interface MorphosCoreStatus {
  dimension: number;
  samples_received: number;
  steps_processed: number;
  wal_records: number;
  dropped_samples: number;
  dual_calibrated: boolean;
  dual_threshold: number | null;
  neural_steps: number;
  breaker_state: string;
  breaker_trips: number;
  temporal_samples: number;
  mode: string;
  chaos_dropped: number;
  latest: {
    step: number;
    timestamp: number;
    actual: number[];
    predicted: number[];
    residual: number;
    residual_ema: number;
    rate: number;
    confidence: number;
    flags: number;
    anomaly: boolean;
    anomaly_score: number | null;
    anomaly_threshold: number | null;
    breaker_state: string;
  } | null;
  branches: Array<{
    name: string;
    steps: number;
    latest_decoded: number[] | null;
    latest_residual: number | null;
    latest_rate: number | null;
  }>;
}

export interface Telemetry {
  type: "telemetry";
  connected: boolean;
  ipc_latency_ms: number | null;
  core: MorphosCoreStatus | null;
  sources: { lorenz: boolean; system: boolean };
  system: number[] | null;
  geometry?: {
    basis_cross_talk: number;
    lorenz_energy: number | null;
    system_energy: number | null;
  };
  system_error?: string | null;
  observed_at: number | null;
  error: string | null;
}

export interface MorphosEvent {
  id: number;
  run_id: number | null;
  sim_t: number | null;
  wall: number;
  level: string;
  message: string;
  data: Record<string, unknown> | null;
}
