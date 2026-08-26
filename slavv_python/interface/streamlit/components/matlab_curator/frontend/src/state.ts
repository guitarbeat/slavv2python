import type {
  AddedEdge,
  CurationSession,
  CuratorSnapshot,
  HistoryEntry,
} from "./types";

export function snapshot(session: CurationSession): CuratorSnapshot {
  return {
    stage: session.stage,
    view: structuredClone(session.view),
    vertex_truth: [...session.vertex_truth],
    vertex_deleted: [...session.vertex_deleted],
    edge_truth: [...session.edge_truth],
    edge_deleted: [...session.edge_deleted],
    added_vertices: structuredClone(session.added_vertices),
    added_edges: structuredClone(session.added_edges),
  };
}

function restore(
  session: CurationSession,
  value: CuratorSnapshot,
  history: HistoryEntry[],
  cursor: number,
): CurationSession {
  return {
    ...session,
    ...structuredClone(value),
    history,
    cursor,
  };
}

export function commit(
  session: CurationSession,
  label: string,
  change: (draft: CurationSession) => void,
): CurationSession {
  const before = snapshot(session);
  const draft = structuredClone(session);
  draft.history = session.history.slice(0, session.cursor);
  change(draft);
  const after = snapshot(draft);
  const entry: HistoryEntry = { label, before, after };
  return {
    ...draft,
    history: [...draft.history, entry].slice(-200),
    cursor: Math.min(draft.history.length + 1, 200),
  };
}

export function undo(session: CurationSession): CurationSession {
  if (session.cursor <= 0) return session;
  const entry = session.history[session.cursor - 1];
  return restore(session, entry.before, session.history, session.cursor - 1);
}

export function redo(session: CurationSession): CurationSession {
  if (session.cursor >= session.history.length) return session;
  const entry = session.history[session.cursor];
  return restore(session, entry.after, session.history, session.cursor + 1);
}

export function straightTrace(start: number[], end: number[]): number[][] {
  const count = Math.max(
    1,
    Math.ceil(Math.max(...start.map((value, index) => Math.abs(end[index] - value)))),
  );
  return Array.from({ length: count + 1 }, (_, index) => {
    const ratio = index / count;
    return start.map((value, axis) => value + (end[axis] - value) * ratio);
  });
}

export function addEdgeBetween(
  session: CurationSession,
  first: number,
  second: number,
  positions: number[][],
  existingConnections: number[][],
): CurationSession {
  if (first === second || !positions[first] || !positions[second]) return session;
  return commit(session, "Add edge", (draft) => {
    existingConnections.forEach((pair, index) => {
      if (
        (pair[0] === first && pair[1] === second) ||
        (pair[0] === second && pair[1] === first)
      ) {
        draft.edge_deleted[index] = true;
      }
    });
    const added: AddedEdge = {
      connections: [first, second],
      trace: straightTrace(positions[first], positions[second]),
      energy: -1.0e30,
    };
    draft.added_edges.push(added);
    draft.edge_truth.push(true);
    draft.edge_deleted.push(false);
  });
}

export function toUint8(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  if (Array.isArray(value)) return Uint8Array.from(value);
  if (
    typeof value === "object" &&
    value !== null &&
    "data" in value &&
    Array.isArray((value as { data: number[] }).data)
  ) {
    return Uint8Array.from((value as { data: number[] }).data);
  }
  return new Uint8Array();
}

export function toFloat32(value: unknown): Float32Array {
  const bytes = toUint8(value);
  return new Float32Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 4));
}

export function toInt16(value: unknown): Int16Array {
  const bytes = toUint8(value);
  return new Int16Array(bytes.buffer, bytes.byteOffset, Math.floor(bytes.byteLength / 2));
}
