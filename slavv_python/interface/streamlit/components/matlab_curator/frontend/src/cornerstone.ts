import { cache, init, volumeLoader } from "@cornerstonejs/core";

import { toUint8 } from "./state";
import type { CuratorData } from "./types";

let initialized = false;

/**
 * Register the volume in Cornerstone's local cache. The MATLAB-faithful MIP
 * canvas uses a deterministic projection renderer today; the same cached
 * physical volume is the foundation for the later linked 3D viewport.
 */
export async function ensureCornerstoneVolume(data: CuratorData): Promise<void> {
  if (!initialized) {
    init();
    initialized = true;
  }
  const volumeId = `slavv-local:${data.volumeKey}`;
  if (cache.getVolume(volumeId)) return;
  const scalarData = toUint8(data.cornerstoneVolume);
  const [y, x, z] = data.shape;
  const [sy, sx, sz] = data.spacing;
  volumeLoader.createLocalVolume(
    volumeId,
    {
      scalarData,
      dimensions: [x, y, z],
      spacing: [sx, sy, sz],
      origin: [0, 0, 0],
      direction: [1, 0, 0, 0, 1, 0, 0, 0, 1],
      metadata: {
        BitsAllocated: 8,
        BitsStored: 8,
        SamplesPerPixel: 1,
        HighBit: 7,
        PhotometricInterpretation: "MONOCHROME2",
        PixelRepresentation: 0,
        FrameOfReferenceUID: data.volumeKey,
        Modality: "OT",
        ImageOrientationPatient: [1, 0, 0, 0, 1, 0],
        PixelSpacing: [sy, sx],
        Columns: x,
        Rows: y,
        voiLut: [],
        VOILUTFunction: "LINEAR",
      },
      preventCache: false,
    },
  );
}
