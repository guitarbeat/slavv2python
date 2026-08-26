import { describe, expect, it } from "vitest";

import {
  computeMip,
  flatIndex,
  projectPoint,
  projectionSize,
  unprojectPoint,
} from "./geometry";

describe("MATLAB projection geometry", () => {
  it("round-trips zero-based YXZ coordinates for every orthogonal view", () => {
    const point = [2, 3, 4];
    ([0, 1, 2] as const).forEach((axis) => {
      const projected = projectPoint(point, axis);
      expect(unprojectPoint(projected, axis, point[axis])).toEqual(point);
    });
  });

  it("computes deterministic slab MIPs from a YXZ C-order buffer", () => {
    const shape: [number, number, number] = [2, 3, 2];
    const volume = new Uint8Array(shape[0] * shape[1] * shape[2]);
    volume[flatIndex(shape, 0, 1, 0)] = 40;
    volume[flatIndex(shape, 0, 1, 1)] = 90;
    const mip = computeMip(volume, shape, 2, 0, 1);
    expect(projectionSize(shape, 2)).toEqual({ width: 3, height: 2 });
    expect(mip.pixels[1]).toBe(90);
  });
});
