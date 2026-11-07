export const EVENT_NAMES = ['Address', 'Toe-up', 'Mid-backswing', 'Top', 'Mid-downswing', 'Impact', 'Mid-follow-through', 'Finish'] as const;
export type EventName = typeof EVENT_NAMES[number];
export type ManualEvent = { name: EventName; frame_index: number; timestamp_seconds: number; source_timestamp_seconds: number; source: 'manual'; updated_at: string };
export type ManualMeasurement = { name: string; value: number; unit: string; event_names: string[]; frame_indices: number[]; interval_seconds: [number, number]; source: 'manual'; reliability: string; method: string; reference: string };
export type Analysis = {
  id: string; status: 'queued' | 'processing' | 'completed' | 'failed' | 'interrupted'; progress: number;
  error?: string; created_at?: string;
  metadata?: { camera_view?: string; handedness?: string; club?: string; mirrored?: boolean; original_filename?: string };
  video?: { duration_seconds: number; frame_count: number; width: number; height: number; timestamps_seconds?: number[]; source_timestamps_seconds?: number[]; source_start_seconds?: number; sha256?: string };
  model?: { status: string; reason?: string };
  annotation_revision?: number; annotation_history?: { revision: number; updated_at: string; events: ManualEvent[] }[];
  events?: ManualEvent[]; measurements?: ManualMeasurement[]; observations?: unknown[];
};
