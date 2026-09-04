import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { ensureCornerstoneVolume } from "./cornerstone";
import {
  computeMip,
  flatIndex,
  inDepth,
  nearestProjectedIndex,
  pointSegmentDistance,
  projectPoint,
  projectionAxes,
  projectionSize,
  unprojectPoint,
  type Point2,
} from "./geometry";
import {
  addEdgeBetween,
  commit,
  redo,
  toFloat32,
  toInt16,
  toUint8,
  undo,
} from "./state";
import type {
  Axis,
  CurationSession,
  CuratorData,
  Stage,
  ToggleMethod,
  Tool,
  ViewState,
} from "./types";

type Trigger = (name: string, value: unknown) => void;

function currentBounds(session: CurationSession, shape: [number, number, number]) {
  const axis = session.view.axis;
  const half = Math.max(0, session.view.thickness);
  return {
    low: Math.max(0, Math.round(session.view.depth - half)),
    high: Math.min(shape[axis] - 1, Math.round(session.view.depth + half)),
  };
}

function histogram(values: number[], bins = 32): number[] {
  if (!values.length) return Array(bins).fill(0);
  const low = Math.min(...values);
  const high = Math.max(...values);
  const width = Math.max(high - low, Number.EPSILON);
  const counts = Array(bins).fill(0);
  values.forEach((value) => {
    const index = Math.min(bins - 1, Math.floor(((value - low) / width) * bins));
    counts[index] += 1;
  });
  return counts;
}

function Histogram({
  title,
  values,
  color,
  footer,
}: {
  title: string;
  values: number[];
  color: string;
  footer?: React.ReactNode;
}) {
  const counts = useMemo(() => histogram(values), [values]);
  const maximum = Math.max(...counts, 1);
  return (
    <section className="mc-panel mc-histogram">
      <header><strong>{title}</strong><span>{values.length.toLocaleString()} samples</span></header>
      <div className="mc-bars" aria-label={title}>
        {counts.map((count, index) => (
          <i
            key={index}
            style={{ height: `${Math.max(2, (count / maximum) * 100)}%`, background: color }}
          />
        ))}
      </div>
      {footer && <div className="mc-histogram-footer">{footer}</div>}
    </section>
  );
}

function Minimap({
  shape,
  spacing,
  view,
}: {
  shape: [number, number, number];
  spacing: [number, number, number];
  view: ViewState;
}) {
  const physical = shape.map((value, index) => value * spacing[index]);
  const maximum = Math.max(...physical, 1);
  const boxWidth = 150 * (physical[1] / maximum);
  const boxHeight = 115 * (physical[0] / maximum);
  const depth = 52 * (physical[2] / maximum);
  const axisLength = shape[view.axis];
  const start = Math.max(0, view.depth - view.thickness) / Math.max(axisLength, 1);
  const end = Math.min(axisLength, view.depth + view.thickness + 1) / Math.max(axisLength, 1);
  return (
    <section className="mc-panel mc-minimap">
      <header><strong>Volume map</strong><span>µm-calibrated FOV</span></header>
      <svg viewBox="0 0 230 170" role="img" aria-label="Physical volume and field of view">
        <g transform="translate(35 32)">
          <path
            d={`M0,${depth} L${boxWidth},${depth} L${boxWidth + depth},0 L${depth},0 Z
                M0,${depth} L0,${boxHeight + depth} L${boxWidth},${boxHeight + depth} L${boxWidth},${depth}
                M${boxWidth},${boxHeight + depth} L${boxWidth + depth},${boxHeight} L${boxWidth + depth},0`}
            className="mc-volume-wire"
          />
          <rect
            x={view.axis === 1 ? boxWidth * start : 0}
            y={view.axis === 0 ? boxHeight * start + depth : depth * start}
            width={view.axis === 1 ? Math.max(4, boxWidth * (end - start)) : boxWidth}
            height={view.axis === 0 ? Math.max(4, boxHeight * (end - start)) : view.axis === 2 ? Math.max(4, depth * (end - start) + 8) : boxHeight}
            className="mc-fov"
          />
        </g>
      </svg>
      <div className="mc-coordinates">
        <span>Y {physical[0].toFixed(0)}</span>
        <span>X {physical[1].toFixed(0)}</span>
        <span>Z {physical[2].toFixed(0)} µm</span>
      </div>
    </section>
  );
}

interface CanvasProps {
  data: CuratorData;
  session: CurationSession;
  tool: Tool;
  method: ToggleMethod;
  positions: number[][];
  connections: number[][];
  traces: number[][][];
  vertexEnergies: number[];
  edgeEnergies: number[];
  onGesture: (start: Point2, end: Point2) => void;
  onView: (change: Partial<ViewState>) => void;
}

function ProjectionCanvas({
  data,
  session,
  tool,
  method,
  positions,
  connections,
  traces,
  vertexEnergies,
  edgeEnergies,
  onGesture,
  onView,
}: CanvasProps) {
  const imageRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const startRef = useRef<Point2 | null>(null);
  const panRef = useRef<{ start: Point2; panX: number; panY: number } | null>(null);
  const volume = useMemo(() => toUint8(data.displayVolume), [data.displayVolume]);
  const bounds = currentBounds(session, data.shape);
  const projection = useMemo(
    () => computeMip(volume, data.shape, session.view.axis, bounds.low, bounds.high),
    [volume, data.shape, session.view.axis, bounds.low, bounds.high],
  );
  const zoom = Math.max(0.5, Math.min(8, session.view.zoom ?? 1));
  const panX = session.view.panX ?? 0;
  const panY = session.view.panY ?? 0;

  useEffect(() => {
    const canvas = imageRef.current;
    if (!canvas) return;
    canvas.width = projection.width;
    canvas.height = projection.height;
    const context = canvas.getContext("2d");
    if (!context) return;
    const offscreen = document.createElement("canvas");
    offscreen.width = projection.width;
    offscreen.height = projection.height;
    const offscreenContext = offscreen.getContext("2d");
    if (!offscreenContext) return;
    const image = offscreenContext.createImageData(projection.width, projection.height);
    const lower = session.view.intensityMin ?? 0;
    const upper = Math.max(lower + 1, session.view.intensityMax ?? 255);
    projection.pixels.forEach((value, index) => {
      let gray = Math.max(0, Math.min(255, ((value - lower) / (upper - lower)) * 255));
      if (session.view.invert) gray = 255 - gray;
      const offset = index * 4;
      image.data[offset] = gray;
      image.data[offset + 1] = gray;
      image.data[offset + 2] = gray;
      image.data[offset + 3] = 255;
    });
    offscreenContext.putImageData(image, 0, 0);
    context.fillStyle = "#030909";
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.save();
    context.translate(canvas.width / 2 + panX, canvas.height / 2 + panY);
    context.scale(zoom, zoom);
    context.imageSmoothingEnabled = false;
    context.drawImage(offscreen, -canvas.width / 2, -canvas.height / 2);
    context.restore();
  }, [projection, session.view.intensityMin, session.view.intensityMax, session.view.invert, zoom, panX, panY]);

  useEffect(() => {
    const canvas = overlayRef.current;
    if (!canvas) return;
    canvas.width = projection.width;
    canvas.height = projection.height;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    context.save();
    context.translate(canvas.width / 2 + panX, canvas.height / 2 + panY);
    context.scale(zoom, zoom);
    context.translate(-canvas.width / 2, -canvas.height / 2);
    const finiteEdgeEnergy = edgeEnergies.filter(Number.isFinite);
    const edgeEnergyMinimum = Math.min(...finiteEdgeEnergy, -1);
    const edgeEnergyMaximum = Math.max(...finiteEdgeEnergy, 0);
    const finiteVertexEnergy = vertexEnergies.filter(Number.isFinite);
    const vertexEnergyMinimum = Math.min(...finiteVertexEnergy, -1);
    const vertexEnergyMaximum = Math.max(...finiteVertexEnergy, 0);
    if (session.stage === "edges") {
      traces.forEach((trace, index) => {
        if (session.edge_deleted[index] || !trace.some((point) => inDepth(point, session.view.axis, bounds.low, bounds.high))) return;
        const rejected =
          !session.edge_truth[index] ||
          (connections[index] ?? []).some(
            (vertex) =>
              vertex < session.vertex_truth.length &&
              (!session.vertex_truth[vertex] || session.vertex_deleted[vertex]),
          );
        const energyFraction =
          edgeEnergyMaximum === edgeEnergyMinimum
            ? 1
            : Math.max(0.15, Math.min(1, ((edgeEnergies[index] ?? edgeEnergyMaximum) - edgeEnergyMinimum) / (edgeEnergyMaximum - edgeEnergyMinimum)));
        context.strokeStyle = rejected ? "#ff565f" : "#35d0ba";
        context.globalAlpha = session.view.binary || rejected ? 1 : 1 - energyFraction * 0.65;
        context.lineWidth = rejected ? 3 / zoom : 2 / zoom;
        context.beginPath();
        trace.forEach((point, pointIndex) => {
          const projected = projectPoint(point, session.view.axis);
          if (pointIndex === 0) context.moveTo(projected.x, projected.y);
          else context.lineTo(projected.x, projected.y);
        });
        context.stroke();
        context.globalAlpha = 1;
      });
    }
    positions.forEach((position, index) => {
      if (
        session.vertex_deleted[index] ||
        !inDepth(position, session.view.axis, bounds.low, bounds.high)
      ) return;
      const point = projectPoint(position, session.view.axis);
      const retained = session.vertex_truth[index];
      const energyFraction =
        vertexEnergyMaximum === vertexEnergyMinimum
          ? 1
          : Math.max(0.15, Math.min(1, ((vertexEnergies[index] ?? vertexEnergyMaximum) - vertexEnergyMinimum) / (vertexEnergyMaximum - vertexEnergyMinimum)));
      context.globalAlpha = session.view.binary || !retained ? 1 : 1 - energyFraction * 0.65;
      context.fillStyle =
        session.stage === "edges" ? "#f2f5f4" : retained ? "#32d6e6" : "#ff565f";
      context.strokeStyle = retained ? "#071111" : "#fff";
      context.lineWidth = 1 / zoom;
      context.beginPath();
      context.arc(point.x, point.y, (retained ? 3.5 : 4.5) / zoom, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.globalAlpha = 1;
    });
    context.restore();

    const axes = projectionAxes(session.view.axis);
    const physicalWidth = projection.width * data.spacing[axes.horizontal] / zoom;
    const target = physicalWidth / 4;
    const magnitude = 10 ** Math.floor(Math.log10(Math.max(target, 1)));
    const barMicrons = Math.max(magnitude, Math.round(target / magnitude) * magnitude);
    const barPixels = (barMicrons / data.spacing[axes.horizontal]) * zoom;
    context.strokeStyle = "#ffffff";
    context.fillStyle = "#ffffff";
    context.lineWidth = 2;
    context.beginPath();
    context.moveTo(16, canvas.height - 18);
    context.lineTo(16 + Math.min(barPixels, canvas.width / 2), canvas.height - 18);
    context.stroke();
    context.font = "11px Inter, sans-serif";
    context.fillText(`${barMicrons.toFixed(0)} µm`, 16, canvas.height - 24);
  }, [data.spacing, positions, traces, connections, vertexEnergies, edgeEnergies, session, projection, bounds.low, bounds.high, zoom, panX, panY]);

  const toVoxel = useCallback(
    (event: React.PointerEvent<HTMLCanvasElement>): Point2 => {
      const rect = event.currentTarget.getBoundingClientRect();
      const px = ((event.clientX - rect.left) / rect.width) * projection.width;
      const py = ((event.clientY - rect.top) / rect.height) * projection.height;
      return {
        x: (px - projection.width / 2 - panX) / zoom + projection.width / 2,
        y: (py - projection.height / 2 - panY) / zoom + projection.height / 2,
      };
    },
    [projection.width, projection.height, panX, panY, zoom],
  );

  return (
    <div className={`mc-canvas-wrap tool-${tool}`}>
      <canvas ref={imageRef} className="mc-image-canvas" />
      <canvas
        ref={overlayRef}
        className="mc-overlay-canvas"
        onContextMenu={(event) => event.preventDefault()}
        onWheel={(event) => {
          event.preventDefault();
          onView({ zoom: Math.max(0.5, Math.min(8, zoom * (event.deltaY < 0 ? 1.15 : 0.87))) });
        }}
        onPointerDown={(event) => {
          event.currentTarget.setPointerCapture(event.pointerId);
          const point = toVoxel(event);
          if (event.button === 1 || event.button === 2 || tool === "view") {
            panRef.current = { start: { x: event.clientX, y: event.clientY }, panX, panY };
          } else {
            startRef.current = point;
          }
        }}
        onPointerMove={(event) => {
          if (!panRef.current) return;
          const rect = event.currentTarget.getBoundingClientRect();
          onView({
            panX: panRef.current.panX + ((event.clientX - panRef.current.start.x) / rect.width) * projection.width,
            panY: panRef.current.panY + ((event.clientY - panRef.current.start.y) / rect.height) * projection.height,
          });
        }}
        onPointerUp={(event) => {
          if (panRef.current) {
            panRef.current = null;
            return;
          }
          const start = startRef.current;
          startRef.current = null;
          if (start) onGesture(start, toVoxel(event));
        }}
      />
      <div className="mc-axis-label mc-axis-x">{projectionAxes(session.view.axis).xLabel}</div>
      <div className="mc-axis-label mc-axis-y">{projectionAxes(session.view.axis).yLabel}</div>
      <div className="mc-canvas-status">
        slices {bounds.low}–{bounds.high} · {method === "circle" ? "circular complement" : method}
      </div>
    </div>
  );
}

export function App({ data, setTriggerValue }: { data: CuratorData; setTriggerValue: Trigger }) {
  const [session, setSession] = useState<CurationSession>(() => structuredClone(data.session));
  const [tool, setTool] = useState<Tool>("toggle");
  const [method, setMethod] = useState<ToggleMethod>("rect");
  const [message, setMessage] = useState("Vertex curation ready");
  const [edgeStart, setEdgeStart] = useState<number | null>(null);
  const [cropPoints, setCropPoints] = useState<number[][]>([]);
  const loadRef = useRef<HTMLInputElement>(null);
  const displayVolume = useMemo(() => toUint8(data.displayVolume), [data.displayVolume]);
  const energyVolume = useMemo(() => toFloat32(data.energyVolume), [data.energyVolume]);
  const scaleVolume = useMemo(() => toInt16(data.scaleVolume), [data.scaleVolume]);
  const sampledIntensities = useMemo(() => {
    const step = Math.max(1, Math.floor(displayVolume.length / 20000));
    const values: number[] = [];
    for (let index = 0; index < displayVolume.length; index += step) {
      values.push(displayVolume[index]);
    }
    return values;
  }, [displayVolume]);

  useEffect(() => {
    setSession(structuredClone(data.session));
    setEdgeStart(null);
    setCropPoints([]);
  }, [data.volumeKey, data.session.baseline_signature, data.sessionRevision]);

  useEffect(() => {
    ensureCornerstoneVolume(data)
      .then(() => setMessage((current) => current.includes("failed") ? current : "Cornerstone volume cached · curator ready"))
      .catch((error) => setMessage(`Cornerstone cache failed; MATLAB projection remains available: ${String(error)}`));
  }, [data.volumeKey, data.cornerstoneVolume, data.shape, data.spacing]);

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setTool("view");
        setEdgeStart(null);
        setCropPoints([]);
        setMessage("Tool stopped");
      } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "z") {
        event.preventDefault();
        setSession((current) => event.shiftKey ? redo(current) : undo(current));
      } else if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "y") {
        event.preventDefault();
        setSession((current) => redo(current));
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, []);

  const positions = useMemo(
    () => [...data.vertices.positions, ...session.added_vertices.map((item) => item.position)],
    [data.vertices.positions, session.added_vertices],
  );
  const energies = useMemo(
    () => [...data.vertices.energies, ...session.added_vertices.map((item) => item.energy)],
    [data.vertices.energies, session.added_vertices],
  );
  const connections = useMemo(
    () => [...data.edges.connections, ...session.added_edges.map((item) => item.connections)],
    [data.edges.connections, session.added_edges],
  );
  const traces = useMemo(
    () => [...data.edges.traces, ...session.added_edges.map((item) => item.trace)],
    [data.edges.traces, session.added_edges],
  );
  const edgeEnergies = useMemo(
    () => [...data.edges.energies, ...session.added_edges.map((item) => item.energy)],
    [data.edges.energies, session.added_edges],
  );
  const bounds = currentBounds(session, data.shape);
  const activeEnergies = session.stage === "vertices" ? energies : edgeEnergies;
  const energyLow = Math.min(...activeEnergies.filter(Number.isFinite), -1);
  const energyHigh = Math.max(...activeEnergies.filter(Number.isFinite), 0);
  const energyThreshold = session.view.energyThreshold ?? energyHigh;

  const updateView = (change: Partial<ViewState>) => {
    setSession((current) => ({ ...current, view: { ...current.view, ...change } }));
  };

  const commitToggle = (start: Point2, end: Point2) => {
    const stage: Stage = session.stage;
    const objectPositions =
      stage === "vertices"
        ? positions
        : traces.map((trace) => trace[Math.floor(trace.length / 2)] ?? [0, 0, 0]);
    const truth = stage === "vertices" ? session.vertex_truth : session.edge_truth;
    const deleted = stage === "vertices" ? session.vertex_deleted : session.edge_deleted;
    const candidates = objectPositions
      .map((position, index) => ({ position, index, point: projectPoint(position, session.view.axis) }))
      .filter(({ position, index }) => !deleted[index] && inDepth(position, session.view.axis, bounds.low, bounds.high));
    let selected: number[] = [];
    const distance = Math.hypot(end.x - start.x, end.y - start.y);
    if (method === "rect") {
      if (distance < 3) {
        const nearest = nearestProjectedIndex(
          objectPositions,
          session.view.axis,
          bounds.low,
          bounds.high,
          end,
          8 / Math.max(session.view.zoom ?? 1, 0.5),
          deleted.map((value) => !value),
        );
        if (nearest !== null) selected = [nearest];
      } else {
        const left = Math.min(start.x, end.x);
        const right = Math.max(start.x, end.x);
        const top = Math.min(start.y, end.y);
        const bottom = Math.max(start.y, end.y);
        selected = candidates
          .filter(({ point }) => point.x >= left && point.x <= right && point.y >= top && point.y <= bottom)
          .map(({ index }) => index);
      }
    } else if (method === "line") {
      selected = candidates
        .filter(({ point }) => pointSegmentDistance(point, start, end) <= 5)
        .map(({ index }) => index);
    } else {
      const radius = distance;
      selected = candidates
        .filter(({ point }) => Math.hypot(point.x - start.x, point.y - start.y) >= radius)
        .map(({ index }) => index);
    }
    if (!selected.length) {
      setMessage("No curatable object was selected");
      return;
    }
    setSession((current) =>
      commit(current, `Toggle ${stage}`, (draft) => {
        const values = stage === "vertices" ? draft.vertex_truth : draft.edge_truth;
        if (selected.length === 1) {
          values[selected[0]] = !values[selected[0]];
        } else {
          const majorityTrue = selected.filter((index) => values[index]).length >= selected.length / 2;
          selected.forEach((index) => {
            values[index] = !majorityTrue;
          });
        }
      }),
    );
    setMessage(`${selected.length} ${stage} toggled`);
  };

  const addVertex = (point: Point2) => {
    if (!data.addVertexAvailable || !energyVolume.length || !scaleVolume.length) {
      setMessage("Add Vertex requires Energy, scale indices, and radius metadata");
      return;
    }
    let bestEnergy = Number.POSITIVE_INFINITY;
    let bestPosition: number[] | null = null;
    let bestScale = 0;
    for (let depth = bounds.low; depth <= bounds.high; depth += 1) {
      const candidate = unprojectPoint(
        { x: Math.round(point.x), y: Math.round(point.y) },
        session.view.axis,
        depth,
      );
      if (candidate.some((value, axis) => value < 0 || value >= data.shape[axis])) continue;
      const index = flatIndex(data.shape, candidate[0], candidate[1], candidate[2]);
      const energy = energyVolume[index];
      if (Number.isFinite(energy) && energy < bestEnergy) {
        bestEnergy = energy;
        bestPosition = candidate;
        bestScale = Math.max(0, scaleVolume[index] ?? 0);
      }
    }
    if (!bestPosition) {
      setMessage("No finite Energy sample was found in this projection slab");
      return;
    }
    const radiusPixelsRaw = data.lumenRadiiPixels[bestScale] ?? 1;
    const radiusPixels = Array.isArray(radiusPixelsRaw) ? radiusPixelsRaw : [radiusPixelsRaw];
    const radiusMicrons = data.lumenRadiiMicrons[bestScale] ?? 1;
    if (
      bestPosition.some((value, axis) => {
        const radius = radiusPixels.length === 1 ? radiusPixels[0] : radiusPixels[axis];
        return value - radius < 0 || value + radius >= data.shape[axis];
      })
    ) {
      setMessage("The proposed vertex would cross the image boundary");
      return;
    }
    setSession((current) =>
      commit(current, "Add vertex", (draft) => {
        draft.added_vertices.push({
          position: bestPosition!,
          energy: bestEnergy,
          scale: bestScale,
          radii_pixels: radiusPixels,
          radius_microns: radiusMicrons,
        });
        draft.vertex_truth.push(true);
        draft.vertex_deleted.push(false);
      }),
    );
    setMessage(`Vertex added at Y${bestPosition[0]}, X${bestPosition[1]}, Z${bestPosition[2]}`);
  };

  const addEdge = (point: Point2) => {
    const allowed = session.vertex_deleted.map((value, index) => !value && session.vertex_truth[index]);
    const selected = nearestProjectedIndex(
      positions,
      session.view.axis,
      bounds.low,
      bounds.high,
      point,
      10 / Math.max(session.view.zoom ?? 1, 0.5),
      allowed,
    );
    if (selected === null) {
      setMessage("Select a retained vertex in the current slab");
      return;
    }
    if (edgeStart === null) {
      setEdgeStart(selected);
      setMessage(`Vertex ${selected} selected · choose the second endpoint`);
      return;
    }
    setSession((current) => addEdgeBetween(current, edgeStart, selected, positions, connections));
    setMessage(`Edge added between vertices ${edgeStart} and ${selected}`);
    setEdgeStart(null);
  };

  const applyCraniumCrop = (centers: number[][]) => {
    const center = [0, 1, 2].map((axis) => centers.reduce((sum, point) => sum + point[axis], 0) / 2);
    const rejected: number[] = [];
    positions.forEach((position, index) => {
      if (session.vertex_deleted[index]) return;
      const vector = position.map((value, axis) => (value - center[axis]) * data.spacing[axis]);
      const distance = Math.hypot(...vector);
      if (!distance) return;
      const radius = data.vertices.radii_microns[index] ?? session.added_vertices[index - data.vertices.positions.length]?.radius_microns ?? 1;
      const before = position.map((value, axis) => Math.round(value - ((radius + 2) * vector[axis]) / distance / data.spacing[axis]));
      const beyond = position.map((value, axis) => Math.round(value + ((radius + 2) * vector[axis]) / distance / data.spacing[axis]));
      const clamp = (point: number[]) => point.map((value, axis) => Math.max(0, Math.min(data.shape[axis] - 1, value)));
      const safeBefore = clamp(before);
      const safeBeyond = clamp(beyond);
      const centerIndex = flatIndex(data.shape, Math.round(position[0]), Math.round(position[1]), Math.round(position[2]));
      const beforeValue = displayVolume[flatIndex(data.shape, safeBefore[0], safeBefore[1], safeBefore[2])] ?? 0;
      const beyondValue = displayVolume[flatIndex(data.shape, safeBeyond[0], safeBeyond[1], safeBeyond[2])] ?? 0;
      const centerValue = displayVolume[centerIndex] ?? 0;
      const first = 2 * beforeValue - 2 * beyondValue;
      const second = 2 * centerValue - beforeValue - beyondValue;
      const feature = first === 0 ? 0 : Math.exp(-second / Math.abs(first));
      if (Number.isFinite(feature) && feature > 0.4) rejected.push(index);
    });
    setSession((current) =>
      commit(current, "Cranium crop", (draft) => {
        rejected.forEach((index) => {
          draft.vertex_truth[index] = false;
        });
      }),
    );
    setCropPoints([]);
    setTool("toggle");
    setMessage(`Cranium crop marked ${rejected.length} vertices false`);
  };

  const handleGesture = (start: Point2, end: Point2) => {
    if (tool === "toggle") commitToggle(start, end);
    else if (tool === "add-vertex") addVertex(end);
    else if (tool === "add-edge") addEdge(end);
    else if (tool === "crop") {
      const point = unprojectPoint(end, session.view.axis, 0);
      const next = [...cropPoints, point];
      setCropPoints(next);
      if (next.length === 3) applyCraniumCrop(next);
      else {
        const nextAxis = ([2, 0, 1] as Axis[])[next.length];
        updateView({
          axis: nextAxis,
          depth: Math.floor(data.shape[nextAxis] / 2),
          thickness: Math.max(1, Math.floor(data.shape[nextAxis] / 2)),
        });
        setMessage(`Crop center ${next.length}/3 selected · mark the next orthogonal view`);
      }
    }
  };

  const applyThreshold = () => {
    const stage = session.stage;
    const objectPositions =
      stage === "vertices"
        ? positions
        : traces.map((trace) => trace[Math.floor(trace.length / 2)] ?? [0, 0, 0]);
    setSession((current) =>
      commit(current, `Threshold ${stage}`, (draft) => {
        const truth = stage === "vertices" ? draft.vertex_truth : draft.edge_truth;
        const values = stage === "vertices" ? energies : edgeEnergies;
        objectPositions.forEach((position, index) => {
          if (inDepth(position, current.view.axis, bounds.low, bounds.high)) {
            truth[index] = (values[index] ?? Number.POSITIVE_INFINITY) < energyThreshold;
          }
        });
      }),
    );
    setMessage(`Local ${stage} threshold applied to slices ${bounds.low}–${bounds.high}`);
  };

  const sweep = () => {
    setSession((current) =>
      commit(current, `Sweep ${current.stage}`, (draft) => {
        const truth = current.stage === "vertices" ? draft.vertex_truth : draft.edge_truth;
        const deleted = current.stage === "vertices" ? draft.vertex_deleted : draft.edge_deleted;
        truth.forEach((value, index) => {
          if (!value) deleted[index] = true;
        });
      }),
    );
    setMessage(`False ${session.stage} swept from the display`);
  };

  const paint = () => {
    const objectPositions = traces.map((trace) => trace[Math.floor(trace.length / 2)] ?? [0, 0, 0]);
    setSession((current) =>
      commit(current, "Paint edges", (draft) => {
        objectPositions.forEach((position, index) => {
          if (!draft.edge_deleted[index] && inDepth(position, current.view.axis, bounds.low, bounds.high)) {
            draft.edge_truth[index] = true;
          }
        });
      }),
    );
    setMessage("Available edges repainted in the current field of view");
  };

  const saveSession = () => {
    const blob = new Blob([JSON.stringify(session, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${session.dataset_name.replace(/[^a-z0-9_-]+/gi, "_")}.slavv-curation.json`;
    anchor.click();
    URL.revokeObjectURL(url);
    setTriggerValue("save", session);
  };

  const counts = {
    retainedVertices: session.vertex_truth.filter((value, index) => value && !session.vertex_deleted[index]).length,
    retainedEdges: session.edge_truth.filter((value, index) => value && !session.edge_deleted[index]).length,
    pending: session.vertex_truth.filter((value) => !value).length + session.edge_truth.filter((value) => !value).length,
  };
  const axis = session.view.axis;
  const axisName = (["Y", "X", "Z"] as const)[axis];
  const maxDepth = data.shape[axis] - 1;

  return (
    <main className="matlab-curator">
      <header className="mc-header">
        <div>
          <span className="mc-eyebrow">
            {data.showTrustClaim
              ? "Trust path · MATLAB-familiar"
              : "SLAVV graphical curator (degraded — Trust claim suppressed)"}
          </span>
          <h2>{session.stage === "vertices" ? "Vertex Curator" : "Edge Curator"}</h2>
        </div>
        <div className="mc-stage-switch" aria-label="Curation stage">
          <button className={session.stage === "vertices" ? "active" : ""} onClick={() => setSession((current) => ({ ...current, stage: "vertices" }))}>1 · Vertices</button>
          <button className={session.stage === "edges" ? "active" : ""} onClick={() => setSession((current) => ({ ...current, stage: "edges" }))}>2 · Edges</button>
        </div>
        <div className="mc-counts">
          <span><b>{counts.retainedVertices}</b> vertices</span>
          <span><b>{counts.retainedEdges}</b> edges</span>
          <span className={counts.pending ? "warning" : ""}><b>{counts.pending}</b> false</span>
        </div>
      </header>

      <div className="mc-workspace">
        <section className="mc-display">
          <div className="mc-display-topbar">
            <div className="mc-axis-buttons">
              {([2, 0, 1] as Axis[]).map((value) => (
                <button
                  key={value}
                  className={axis === value ? "active" : ""}
                  onClick={() => updateView({ axis: value, depth: Math.floor(data.shape[value] / 2), thickness: Math.max(1, Math.floor(data.shape[value] / 8)), panX: 0, panY: 0 })}
                >
                  {(["Y", "X", "Z"] as const)[value]} projection
                </button>
              ))}
            </div>
            <button className="mc-quiet" onClick={() => updateView({ zoom: 1, panX: 0, panY: 0 })}>Reset view</button>
          </div>
          <ProjectionCanvas
            data={data}
            session={session}
            tool={tool}
            method={method}
            positions={positions}
            connections={connections}
            traces={traces}
            vertexEnergies={energies}
            edgeEnergies={edgeEnergies}
            onGesture={handleGesture}
            onView={updateView}
          />
          <div className="mc-slice-controls">
            <label>
              <span>{axisName}-Depth <b>{session.view.depth}</b></span>
              <input type="range" min={0} max={maxDepth} value={session.view.depth} onChange={(event) => updateView({ depth: Number(event.target.value) })} />
            </label>
            <label>
              <span>{axisName}-Thickness <b>{session.view.thickness}</b></span>
              <input type="range" min={0} max={Math.max(1, Math.floor(maxDepth / 2))} value={session.view.thickness} onChange={(event) => updateView({ thickness: Number(event.target.value) })} />
            </label>
          </div>
          <div className="mc-tool-ribbon">
            <button onClick={() => setSession((current) => undo(current))} disabled={session.cursor === 0}>Undo</button>
            <button onClick={() => setSession((current) => redo(current))} disabled={session.cursor >= session.history.length}>Redo</button>
            {session.stage === "vertices" ? (
              <button className={tool === "crop" ? "active" : ""} disabled={!data.originalAvailable} onClick={() => { setTool("crop"); setCropPoints([]); updateView({ axis: 2, depth: Math.floor(data.shape[2] / 2), thickness: Math.max(1, Math.floor(data.shape[2] / 2)) }); setMessage("Crop 0/3 · mark the cranium center in the Z projection"); }}>Crop</button>
            ) : (
              <button onClick={paint}>Paint</button>
            )}
            <button onClick={sweep}>Sweep</button>
            <button className={tool === "add-vertex" ? "active" : ""} disabled={!data.addVertexAvailable} onClick={() => { setTool("add-vertex"); setMessage("Click the projection to add a vertex at the local Energy minimum"); }}>Add Vertex</button>
            {session.stage === "edges" && <button className={tool === "add-edge" ? "active" : ""} onClick={() => { setTool("add-edge"); setEdgeStart(null); setMessage("Choose two retained vertices"); }}>Add Edge</button>}
            <button className={tool === "toggle" ? "active primary" : "primary"} onClick={() => setTool("toggle")}>Toggle</button>
            <select value={method} onChange={(event) => setMethod(event.target.value as ToggleMethod)} aria-label="Toggle method">
              <option value="rect">rect</option>
              <option value="line">line</option>
              <option value="circle">circ. comp.</option>
            </select>
          </div>
        </section>

        <aside className="mc-context">
          <Minimap shape={data.shape} spacing={data.spacing} view={session.view} />
          <Histogram
            title="Intensity histogram"
            values={sampledIntensities}
            color="#dce7e5"
            footer={
              <>
                <label>Min <input type="number" min={0} max={254} value={session.view.intensityMin ?? 0} onChange={(event) => updateView({ intensityMin: Number(event.target.value) })} /></label>
                <button onClick={() => updateView({ invert: !session.view.invert })}>{session.view.invert ? "Inverted" : "Original"}</button>
                <label>Max <input type="number" min={1} max={255} value={session.view.intensityMax ?? 255} onChange={(event) => updateView({ intensityMax: Number(event.target.value) })} /></label>
              </>
            }
          />
          <Histogram
            title={session.stage === "vertices" ? "Vertex energy" : "Edge energy"}
            values={activeEnergies.filter(Number.isFinite)}
            color="#35d0ba"
            footer={
              <div className="mc-threshold">
                <span>{energyLow.toPrecision(3)}</span>
                <input type="range" min={energyLow} max={energyHigh} step={Math.max((energyHigh - energyLow) / 200, 0.00001)} value={energyThreshold} onChange={(event) => updateView({ energyThreshold: Number(event.target.value) })} />
                <span>{energyHigh.toPrecision(3)}</span>
                <button onClick={applyThreshold}>Threshold {energyThreshold.toPrecision(3)}</button>
                <button onClick={() => updateView({ binary: !session.view.binary })}>{session.view.binary ? "Binary" : "Graded"}</button>
              </div>
            }
          />
        </aside>
      </div>

      <footer className="mc-footer">
        <div className="mc-status"><i /><span>{message}</span>{data.degradedReason && <em>{data.degradedReason}</em>}</div>
        <div className="mc-commit">
          <input
            ref={loadRef}
            type="file"
            accept=".json,.slavv-curation.json,application/json"
            hidden
            onChange={async (event) => {
              const file = event.target.files?.[0];
              if (!file) return;
              try {
                setTriggerValue("load", JSON.parse(await file.text()));
              } catch {
                setMessage("The selected curation file is not valid JSON");
              }
              event.target.value = "";
            }}
          />
          <button onClick={() => loadRef.current?.click()}>Load</button>
          <button onClick={saveSession}>Save</button>
          {session.stage === "vertices" ? (
            <button className="primary" onClick={() => { setSession((current) => ({ ...current, stage: "edges" })); setTool("toggle"); setMessage("Vertex state carried forward · edge curation ready"); }}>Continue to edges</button>
          ) : (
            <button className="primary" onClick={() => setTriggerValue("apply", session)}>Apply and rebuild network</button>
          )}
        </div>
      </footer>
    </main>
  );
}
