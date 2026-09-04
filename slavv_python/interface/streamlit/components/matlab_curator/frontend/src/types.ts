export type Axis = 0 | 1 | 2;
export type Stage = "vertices" | "edges";
export type Tool =
  | "view"
  | "toggle"
  | "add-vertex"
  | "add-edge"
  | "crop";
export type ToggleMethod = "rect" | "line" | "circle";

export interface AddedVertex {
  position: number[];
  energy: number;
  scale: number;
  radii_pixels: number[];
  radius_microns: number;
}

export interface AddedEdge {
  connections: number[];
  trace: number[][];
  energy: number;
}

export interface ViewState {
  axis: Axis;
  depth: number;
  thickness: number;
  invert: boolean;
  binary: boolean;
  intensityMin?: number;
  intensityMax?: number;
  energyThreshold?: number;
  zoom?: number;
  panX?: number;
  panY?: number;
}

export interface CuratorSnapshot {
  stage: Stage;
  view: ViewState;
  vertex_truth: boolean[];
  vertex_deleted: boolean[];
  edge_truth: boolean[];
  edge_deleted: boolean[];
  added_vertices: AddedVertex[];
  added_edges: AddedEdge[];
}

export interface HistoryEntry {
  label: string;
  before: CuratorSnapshot;
  after: CuratorSnapshot;
}

export interface CurationSession extends CuratorSnapshot {
  schema_version: 1;
  baseline_signature: string;
  dataset_name: string;
  history: HistoryEntry[];
  cursor: number;
}

export interface CuratorVertices {
  positions: number[][];
  energies: number[];
  scales: number[];
  radii_pixels: number[] | number[][];
  radii_microns: number[];
}

export interface CuratorEdges {
  traces: number[][][];
  connections: number[][];
  energies: number[];
}

export interface CuratorData {
  volumeKey: string;
  sessionRevision: number;
  displayVolume: unknown;
  cornerstoneVolume: unknown;
  energyVolume?: unknown;
  scaleVolume?: unknown;
  shape: [number, number, number];
  spacing: [number, number, number];
  displayRange: [number, number];
  originalAvailable: boolean;
  addVertexAvailable: boolean;
  degradedReason?: string;
  showTrustClaim?: boolean;
  vertices: CuratorVertices;
  edges: CuratorEdges;
  lumenRadiiPixels: number[] | number[][];
  lumenRadiiMicrons: number[];
  session: CurationSession;
}
