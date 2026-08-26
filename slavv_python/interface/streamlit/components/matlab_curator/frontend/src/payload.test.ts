import { describe, expect, it } from "vitest";

import { decodeCuratorPayload } from "./payload";

function packedPayload(): Uint8Array {
  const buffers = [
    new Uint8Array([1, 2]),
    new Uint8Array([2, 1]),
    new Uint8Array(new Float32Array([-1, -2]).buffer),
    new Uint8Array(new Int16Array([0, 1]).buffer),
  ];
  let offset = 0;
  const descriptors: Record<string, object> = {};
  ["displayVolume", "cornerstoneVolume", "energyVolume", "scaleVolume"].forEach(
    (name, index) => {
      descriptors[name] = { dtype: "fixture", offset, length: buffers[index].byteLength };
      offset += buffers[index].byteLength;
    },
  );
  const header = new TextEncoder().encode(JSON.stringify({
    data: {
      volumeKey: "fixture",
      sessionRevision: 0,
      shape: [1, 1, 2],
      spacing: [1, 1, 1],
      displayRange: [0, 1],
      originalAvailable: true,
      addVertexAvailable: true,
      vertices: { positions: [], energies: [], scales: [], radii_pixels: [], radii_microns: [] },
      edges: { traces: [], connections: [], energies: [] },
      lumenRadiiPixels: [1],
      lumenRadiiMicrons: [1],
    },
    session: {
      schema_version: 1,
      baseline_signature: "fixture",
      dataset_name: "fixture",
      stage: "vertices",
      view: { axis: 2, depth: 0, thickness: 0, invert: true, binary: false },
      vertex_truth: [], vertex_deleted: [], edge_truth: [], edge_deleted: [],
      added_vertices: [], added_edges: [], history: [], cursor: 0,
    },
    buffers: descriptors,
  }));
  const result = new Uint8Array(4 + header.byteLength + offset);
  new DataView(result.buffer).setUint32(0, header.byteLength, true);
  result.set(header, 4);
  let bodyOffset = 4 + header.byteLength;
  buffers.forEach((buffer) => {
    result.set(buffer, bodyOffset);
    bodyOffset += buffer.byteLength;
  });
  return result;
}

describe("curator binary contract", () => {
  it("decodes typed volume buffers without Arrow or base64 conversion", () => {
    const data = decodeCuratorPayload(packedPayload());
    expect(Array.from(data.displayVolume as Uint8Array)).toEqual([1, 2]);
    expect(Array.from(data.cornerstoneVolume as Uint8Array)).toEqual([2, 1]);
    expect(Array.from(data.energyVolume as Float32Array)).toEqual([-1, -2]);
    expect(Array.from(data.scaleVolume as Int16Array)).toEqual([0, 1]);
  });
});
