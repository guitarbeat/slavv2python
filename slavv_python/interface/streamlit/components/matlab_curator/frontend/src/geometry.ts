import type { Axis } from "./types";

export interface Point2 {
  x: number;
  y: number;
}

export function projectionAxes(axis: Axis): {
  vertical: Axis;
  horizontal: Axis;
  xLabel: string;
  yLabel: string;
} {
  if (axis === 2) return { vertical: 0, horizontal: 1, xLabel: "X", yLabel: "Y" };
  if (axis === 0) return { vertical: 1, horizontal: 2, xLabel: "Z", yLabel: "X" };
  return { vertical: 0, horizontal: 2, xLabel: "Z", yLabel: "Y" };
}

export function projectionSize(
  shape: [number, number, number],
  axis: Axis,
): { width: number; height: number } {
  const axes = projectionAxes(axis);
  return { width: shape[axes.horizontal], height: shape[axes.vertical] };
}

export function projectPoint(point: number[], axis: Axis): Point2 {
  const axes = projectionAxes(axis);
  return { x: point[axes.horizontal], y: point[axes.vertical] };
}

export function unprojectPoint(
  point: Point2,
  axis: Axis,
  depth: number,
): [number, number, number] {
  const result: [number, number, number] = [0, 0, 0];
  const axes = projectionAxes(axis);
  result[axis] = depth;
  result[axes.horizontal] = point.x;
  result[axes.vertical] = point.y;
  return result;
}

export function inDepth(
  point: number[],
  axis: Axis,
  low: number,
  high: number,
): boolean {
  return point[axis] >= low && point[axis] <= high;
}

export function flatIndex(
  shape: [number, number, number],
  y: number,
  x: number,
  z: number,
): number {
  return (y * shape[1] + x) * shape[2] + z;
}

export function pointSegmentDistance(
  point: Point2,
  start: Point2,
  end: Point2,
): number {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  if (dx === 0 && dy === 0) return Math.hypot(point.x - start.x, point.y - start.y);
  const ratio = Math.max(
    0,
    Math.min(1, ((point.x - start.x) * dx + (point.y - start.y) * dy) / (dx * dx + dy * dy)),
  );
  return Math.hypot(point.x - (start.x + ratio * dx), point.y - (start.y + ratio * dy));
}

export function computeMip(
  volume: Uint8Array,
  shape: [number, number, number],
  axis: Axis,
  low: number,
  high: number,
): { pixels: Uint8Array; width: number; height: number } {
  const { width, height } = projectionSize(shape, axis);
  const pixels = new Uint8Array(width * height);
  for (let v = 0; v < height; v += 1) {
    for (let u = 0; u < width; u += 1) {
      let maximum = 0;
      for (let depth = low; depth <= high; depth += 1) {
        const coordinates = unprojectPoint({ x: u, y: v }, axis, depth);
        const index = flatIndex(shape, coordinates[0], coordinates[1], coordinates[2]);
        maximum = Math.max(maximum, volume[index] ?? 0);
      }
      pixels[v * width + u] = maximum;
    }
  }
  return { pixels, width, height };
}

export function nearestProjectedIndex(
  positions: number[][],
  axis: Axis,
  low: number,
  high: number,
  point: Point2,
  maximumDistance: number,
  allowed?: boolean[],
): number | null {
  let best: number | null = null;
  let bestDistance = maximumDistance;
  positions.forEach((position, index) => {
    if (!inDepth(position, axis, low, high) || (allowed && !allowed[index])) return;
    const projected = projectPoint(position, axis);
    const distance = Math.hypot(projected.x - point.x, projected.y - point.y);
    if (distance < bestDistance) {
      best = index;
      bestDistance = distance;
    }
  });
  return best;
}
