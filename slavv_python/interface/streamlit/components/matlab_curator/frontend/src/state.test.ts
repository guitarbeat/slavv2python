import { describe, expect, it } from "vitest";

import { addEdgeBetween, commit, redo, straightTrace, undo } from "./state";
import type { CurationSession } from "./types";

function session(): CurationSession {
  return {
    schema_version: 1,
    baseline_signature: "fixture",
    dataset_name: "fixture",
    stage: "vertices",
    view: { axis: 2, depth: 1, thickness: 1, invert: true, binary: false },
    vertex_truth: [true, true],
    vertex_deleted: [false, false],
    edge_truth: [true],
    edge_deleted: [false],
    added_vertices: [],
    added_edges: [],
    history: [],
    cursor: 0,
  };
}

describe("reversible curator state", () => {
  it("undoes and redoes exact object patches", () => {
    const changed = commit(session(), "toggle", (draft) => {
      draft.vertex_truth[1] = false;
    });
    expect(changed.vertex_truth).toEqual([true, false]);
    expect(undo(changed).vertex_truth).toEqual([true, true]);
    expect(redo(undo(changed)).vertex_truth).toEqual([true, false]);
  });

  it("uses MATLAB straight L-infinity interpolation and replaces duplicates", () => {
    expect(straightTrace([0, 0, 0], [2, 1, 0])).toEqual([
      [0, 0, 0],
      [1, 0.5, 0],
      [2, 1, 0],
    ]);
    const changed = addEdgeBetween(
      session(),
      0,
      1,
      [[0, 0, 0], [2, 1, 0]],
      [[1, 0]],
    );
    expect(changed.edge_deleted).toEqual([true, false]);
    expect(changed.added_edges).toHaveLength(1);
    expect(changed.added_edges[0].connections).toEqual([0, 1]);
  });
});
