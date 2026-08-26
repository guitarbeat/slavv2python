import type { CuratorData } from "./types";

interface BufferDescriptor {
  dtype: string;
  offset: number;
  length: number;
}

interface PackedHeader {
  data: Omit<CuratorData, "displayVolume" | "cornerstoneVolume" | "energyVolume" | "scaleVolume" | "session">;
  session: CuratorData["session"];
  buffers: Record<string, BufferDescriptor>;
}

function asBytes(value: unknown): Uint8Array {
  if (value instanceof Uint8Array) return value;
  if (value instanceof ArrayBuffer) return new Uint8Array(value);
  if (ArrayBuffer.isView(value)) {
    return new Uint8Array(value.buffer, value.byteOffset, value.byteLength);
  }
  throw new Error("The curator received an unsupported binary payload");
}

function bufferSlice(
  bytes: Uint8Array,
  bodyOffset: number,
  descriptor: BufferDescriptor,
): ArrayBuffer {
  const start = bodyOffset + descriptor.offset;
  const end = start + descriptor.length;
  if (start < bodyOffset || end > bytes.byteLength || end < start) {
    throw new Error("The curator binary payload is truncated");
  }
  return bytes.slice(start, end).buffer;
}

export function decodeCuratorPayload(value: unknown): CuratorData {
  const bytes = asBytes(value);
  if (bytes.byteLength < 4) throw new Error("The curator binary payload is empty");
  const headerLength = new DataView(bytes.buffer, bytes.byteOffset, 4).getUint32(0, true);
  const bodyOffset = 4 + headerLength;
  if (bodyOffset > bytes.byteLength) throw new Error("The curator header is truncated");
  const headerText = new TextDecoder().decode(bytes.subarray(4, bodyOffset));
  const header = JSON.parse(headerText) as PackedHeader;
  const descriptor = (name: string) => {
    const result = header.buffers[name];
    if (!result) throw new Error(`The curator payload is missing ${name}`);
    return result;
  };
  const displayVolume = new Uint8Array(bufferSlice(bytes, bodyOffset, descriptor("displayVolume")));
  const cornerstoneVolume = new Uint8Array(bufferSlice(bytes, bodyOffset, descriptor("cornerstoneVolume")));
  const energyVolume = new Float32Array(bufferSlice(bytes, bodyOffset, descriptor("energyVolume")));
  const scaleVolume = new Int16Array(bufferSlice(bytes, bodyOffset, descriptor("scaleVolume")));
  return {
    ...header.data,
    session: header.session,
    displayVolume,
    cornerstoneVolume,
    energyVolume,
    scaleVolume,
  } as CuratorData;
}
