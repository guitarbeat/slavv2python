# Ground-Truth Architecture Specification: Distance Transforms

- **Subsystem:** Distance Transforms (Geodesic & Euclidean EDT)
- **Native Binary:** `/Volumes/LoveSSD/Applications/MATLAB_R2024b.app/bin/maca64/libmwimagescore.dylib`
- **Primary Function:** `_graydist`
- **Symbol:** `__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_`
- **Instructions Decoded:** 2962 | **Branches:** 399 | **Calls:** 266
- **Corresponding Python Implementation:** [`slavv_python.pipeline.edges.distance`](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/slavv_python/pipeline/edges/distance.py)

---

## 1. Algorithmic Overview

1. **`bwdist` (Euclidean Distance Transform):**
   - Implemented via Saito-Toriwaki / Meijster separable parabolic envelope algorithm.
   - Parallelized across CPU threads via Intel TBB (`__Z13bwdistEDT_tbbPKhPKmmPf`).
   - Integer grid coordinate offsets Delta_x^2 + Delta_y^2 + Delta_z^2 are computed strictly in exact integer arithmetic before a single square root conversion.
2. **`graydist` (Gray-Weighted Geodesic Distance):**
   - Implemented via Fast Marching / Dijkstra-style priority propagation with chamfer weights.
   - Solves the Eikonal equation across voxels.
   - Uses double-precision floating-point accumulators to avoid truncation drift along long vascular strands.

## 2. Disassembly Excerpt (`_graydist`)

```arm64
   c0f24: d10203ff     	sub	sp, sp, #0x80
   c0f28: a90367fa     	stp	x26, x25, [sp, #0x30]
   c0f2c: a9045ff8     	stp	x24, x23, [sp, #0x40]
   c0f30: a90557f6     	stp	x22, x21, [sp, #0x50]
   c0f34: a9064ff4     	stp	x20, x19, [sp, #0x60]
   c0f38: a9077bfd     	stp	x29, x30, [sp, #0x70]
   c0f3c: 9101c3fd     	add	x29, sp, #0x70
   c0f40: aa0103f4     	mov	x20, x1
   c0f44: a9405c73     	ldp	x19, x23, [x3]
   c0f48: a9416478     	ldp	x24, x25, [x3, #0x10]
   c0f4c: b9401275     	ldr	w21, [x19, #0x10]
   c0f50: 710046bf     	cmp	w21, #0x11
   c0f54: 5400008b     	b.lt	0xc0f64 <__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x40>
   c0f58: aa1303e0     	mov	x0, x19
   c0f5c: 9409cf8f     	bl	0x334d98 <dyld_stub_binder+0x334d98>
   c0f60: aa0003f5     	mov	x21, x0
   c0f64: b9402268     	ldr	w8, [x19, #0x20]
   c0f68: 37980088     	tbnz	w8, #0x13, 0xc0f78 <__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x54>
   c0f6c: f9401a7a     	ldr	x26, [x19, #0x30]
   c0f70: 37400088     	tbnz	w8, #0x8, 0xc0f80 <__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x5c>
   c0f74: 14000006     	b	0xc0f8c <__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x68>
   c0f78: 5280005a     	mov	w26, #0x2               ; =2
   c0f7c: 36400088     	tbz	w8, #0x8, 0xc0f8c <__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x68>
   c0f80: aa1303e0     	mov	x0, x19
   c0f84: 9409cf88     	bl	0x334da4 <dyld_stub_binder+0x334da4>
   c0f88: b9402268     	ldr	w8, [x19, #0x20]
   c0f8c: 9100a276     	add	x22, x19, #0x28
   c0f90: 37980048     	tbnz	w8, #0x13, 0xc0f98 <__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x74>
   c0f94: f94002d6     	ldr	x22, [x22]
   c0f98: b9401262     	ldr	w2, [x19, #0x10]
   c0f9c: 7100445f     	cmp	w2, #0x11
   c0fa0: 5400008b     	b.lt	0xc0fb0 <__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x8c>
   c0fa4: aa1303e0     	mov	x0, x19
   c0fa8: 9409cf7c     	bl	0x334d98 <dyld_stub_binder+0x334d98>
   c0fac: aa0003e2     	mov	x2, x0
   c0fb0: 8b1a0ec1     	add	x1, x22, x26, lsl #3
   c0fb4: 910023e8     	add	x8, sp, #0x8
   c0fb8: aa1603e0     	mov	x0, x22
   c0fbc: 52800003     	mov	w3, #0x0                ; =0
   c0fc0: 9409cf4c     	bl	0x334cf0 <dyld_stub_binder+0x334cf0>
```

## 3. Key Callees
- `dyld_stub_binder+0x334d98`
- `dyld_stub_binder+0x334da4`
- `dyld_stub_binder+0x334cf0`
- `__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x144`
- `__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x1e28`
- `dyld_stub_binder+0x334b64`
- `__Z10_applylutciPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0xd1c`
- `dyld_stub_binder+0x3349f0`
- `dyld_stub_binder+0x3349cc`
- `dyld_stub_binder+0x334e94`
- `__Z9_bwlabelniPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0xb34`
- `__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x5e0`
- `dyld_stub_binder+0x334e70`
- `__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x388`
- `dyld_stub_binder+0x33499c`
- `__Z9_bwlabelniPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0xb20`
- `__Z9_bwlabelniPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0xa28`
- `__Z9_bwlabelniPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0xa9c`
- `dyld_stub_binder+0x3348a0`
- `dyld_stub_binder+0x335134`
- `dyld_stub_binder+0x335128`
- `__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x1a00`
- `__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x1714`
- `__Z9_graydistiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x1890`
- `__Z13_fastmarchingiPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0xb04`
