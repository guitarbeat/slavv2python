# Ground-Truth Architecture Specification: Watershed Segmentation

- **Subsystem:** Watershed Segmentation (Meyer & Vincent-Soille)
- **Native Binary:** `/Volumes/LoveSSD/Applications/MATLAB_R2024b.app/bin/maca64/libmwimagescore.dylib`
- **Primary Function:** `_watershed_meyer`
- **Symbol:** `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_`
- **Instructions Decoded:** 13114 | **Branches:** 2515 | **Calls:** 794
- **Corresponding Python Implementation:** [`slavv_python.pipeline.edges.watershed`](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/slavv_python/pipeline/edges/watershed.py)

---

## 1. Algorithmic Overview

MATLAB's `watershed(A, conn)` executes **Meyer's Topological Watershed Algorithm** by default:
1. **Regional Minima Extraction:** Discovers connected minimum catchment basins using `_bwlabeln`.
2. **Hierarchical Priority Queue:** Allocates a multi-bucket queue corresponding to discrete energy levels.
3. **Flooding & Catchment Expansion:**
   - Voxels on the frontier are pushed to their corresponding energy bucket.
   - For continuous floating-point fields, values are quantized or ordered via continuous sorted lists.
   - When adjacent catchments meet, ridge pixels are tagged with label `0` (watershed boundary) or merged if no ridge is requested.

## 2. Priority Queue & Tie-Breaking Rules

> [!IMPORTANT]
> **Lowest Fortran Linear Index Priority:**
> When two frontier voxels possess identical energy values, MATLAB evaluates them in order of their **Fortran-order (column-major) linear index**:
> `i_linear = y + x * Ny + z * (Ny * Nx)`
> Under Fortran order, the Y-coordinate increments fastest, followed by X, then Z.
> This proves the strict necessity of SLAVV's **`[Y, X, Z]` internal grid alignment** (ADR 0012) in Python.

## 3. Disassembly Excerpt (`_watershed_meyer`)

```arm64
  320f88: d106c3ff     	sub	sp, sp, #0x1b0
  320f8c: a9156ffc     	stp	x28, x27, [sp, #0x150]
  320f90: a91667fa     	stp	x26, x25, [sp, #0x160]
  320f94: a9175ff8     	stp	x24, x23, [sp, #0x170]
  320f98: a91857f6     	stp	x22, x21, [sp, #0x180]
  320f9c: a9194ff4     	stp	x20, x19, [sp, #0x190]
  320fa0: a91a7bfd     	stp	x29, x30, [sp, #0x1a0]
  320fa4: 910683fd     	add	x29, sp, #0x1a0
  320fa8: aa0303f8     	mov	x24, x3
  320fac: aa0203e0     	mov	x0, x2
  320fb0: aa0103f9     	mov	x25, x1
  320fb4: d102a3a2     	sub	x2, x29, #0xa8
  320fb8: aa0303e1     	mov	x1, x3
  320fbc: 97fffe8a     	bl	0x3209e4 <__Z12check_inputsiPPK11mxArray_tagPPS_>
  320fc0: f9400314     	ldr	x20, [x24]
  320fc4: b9401288     	ldr	w8, [x20, #0x10]
  320fc8: 71000d1f     	cmp	w8, #0x3
  320fcc: 54000101     	b.ne	0x320fec <__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x64>
  320fd0: aa1403e0     	mov	x0, x20
  320fd4: 94004ef3     	bl	0x334ba0 <dyld_stub_binder+0x334ba0>
  320fd8: aa0003fb     	mov	x27, x0
  320fdc: f9400314     	ldr	x20, [x24]
  320fe0: b9402289     	ldr	w9, [x20, #0x20]
  320fe4: 374000a9     	tbnz	w9, #0x8, 0x320ff8 <__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x70>
  320fe8: 14000007     	b	0x321004 <__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x7c>
  320fec: f940269b     	ldr	x27, [x20, #0x48]
  320ff0: b9402289     	ldr	w9, [x20, #0x20]
  320ff4: 36400089     	tbz	w9, #0x8, 0x321004 <__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x7c>
  320ff8: aa1403e0     	mov	x0, x20
  320ffc: 94004f6a     	bl	0x334da4 <dyld_stub_binder+0x334da4>
  321000: b9402289     	ldr	w9, [x20, #0x20]
  321004: f9401a88     	ldr	x8, [x20, #0x30]
  321008: 37980149     	tbnz	w9, #0x13, 0x321030 <__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0xa8>
  32100c: b4000188     	cbz	x8, 0x32103c <__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0xb4>
  321010: f940168a     	ldr	x10, [x20, #0x28]
  321014: d1000509     	sub	x9, x8, #0x1
  321018: 9240f129     	and	x9, x9, #0x1fffffffffffffff
  32101c: f1000d3f     	cmp	x9, #0x3
  321020: 54000122     	b.hs	0x321044 <__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0xbc>
  321024: 52800034     	mov	w20, #0x1               ; =1
```

## 4. Key Callees
- `__Z12check_inputsiPPK11mxArray_tagPPS_`
- `dyld_stub_binder+0x334ba0`
- `dyld_stub_binder+0x334da4`
- `dyld_stub_binder+0x334d98`
- `dyld_stub_binder+0x334cd8`
- `dyld_stub_binder+0x334b64`
- `dyld_stub_binder+0x3349f0`
- `dyld_stub_binder+0x3349cc`
- `dyld_stub_binder+0x334e94`
- `__Z9_bwlabelniPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0xb34`
- `dyld_stub_binder+0x334984`
- `dyld_stub_binder+0x334e70`
- `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x1a00`
- `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0xaf7c`
- `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x7de8`
- `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0xbdc8`
- `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x4234`
- `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x91e4`
- `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x69ec`
- `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x2e44`
- `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x5624`
- `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x8264`
- `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x1e78`
- `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x46ac`
- `__Z16_watershed_meyeriPNSt3__110unique_ptrI11mxArray_tagN6matrix6detail17mxDestroy_deleterEEEiPPKS1_+0x965c`
