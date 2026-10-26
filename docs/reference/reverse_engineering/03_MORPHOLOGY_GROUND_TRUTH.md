# Ground-Truth Architecture Specification: 3D Morphology

- **Subsystem:** 3D Morphology & Linear Structuring Elements
- **Native Binary:** `/Volumes/LoveSSD/Applications/MATLAB_R2024b.app/bin/maca64/libmwmorphop_flat.dylib`
- **Primary Function:** `is_linear_strel`
- **Symbol:** `__Z15is_linear_strelPKhPKmmPmS3_Pl`
- **Instructions Decoded:** 81 | **Branches:** 16 | **Calls:** 0
- **Corresponding Python Implementation:** [`slavv_python.pipeline.preprocessing.construct_structuring_element`](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/slavv_python/pipeline/preprocessing/construct_structuring_element.py)

---

## 1. Algorithmic Overview

1. **Linear Structuring Element Fast-Path (`is_linear_strel`):**
   - Detects whether a 3D structuring element consists of collinear non-zero voxels.
   - Decomposes arbitrary multi-scale 3D structuring elements into 1D directional passes along principal directions.
2. **Flat Dilation & Erosion (`dilate_flat_real64`, `erode_flat_real64`):**
   - Uses IEEE 754 infinity sentinel initializations:
     - Dilation accumulator initialized to -infinity (`fmov d9, #0.0` / `__Z15init_val_dilateIdET_v`).
     - Erosion accumulator initialized to +infinity (`__Z14init_val_erodeIdET_v`).
   - Eliminates boundary bias by ignoring padded infinity regions during min/max reductions.

## 2. Disassembly Excerpt (`is_linear_strel`)

```arm64
     730: b4000882     	cbz	x2, 0x840 <__Z15is_linear_strelPKhPKmmPmS3_Pl+0x110>
     734: f100105f     	cmp	x2, #0x4
     738: 54000082     	b.hs	0x748 <__Z15is_linear_strelPKhPKmmPmS3_Pl+0x18>
     73c: d2800008     	mov	x8, #0x0                ; =0
     740: 52800029     	mov	w9, #0x1                ; =1
     744: 14000015     	b	0x798 <__Z15is_linear_strelPKhPKmmPmS3_Pl+0x68>
     748: 927ef448     	and	x8, x2, #0xfffffffffffffffc
     74c: 9100402a     	add	x10, x1, #0x10
     750: 52800029     	mov	w9, #0x1                ; =1
     754: aa0803eb     	mov	x11, x8
     758: 5280002c     	mov	w12, #0x1               ; =1
     75c: 5280002d     	mov	w13, #0x1               ; =1
     760: 5280002e     	mov	w14, #0x1               ; =1
     764: a97f414f     	ldp	x15, x16, [x10, #-0x10]
     768: a8c21951     	ldp	x17, x6, [x10], #0x20
     76c: 9b097de9     	mul	x9, x15, x9
     770: 9b0c7e0c     	mul	x12, x16, x12
     774: 9b0d7e2d     	mul	x13, x17, x13
     778: 9b0e7cce     	mul	x14, x6, x14
     77c: f100116b     	subs	x11, x11, #0x4
     780: 54ffff21     	b.ne	0x764 <__Z15is_linear_strelPKhPKmmPmS3_Pl+0x34>
     784: 9b097d89     	mul	x9, x12, x9
     788: 9b097da9     	mul	x9, x13, x9
     78c: 9b097dc9     	mul	x9, x14, x9
     790: eb02011f     	cmp	x8, x2
     794: 540000e0     	b.eq	0x7b0 <__Z15is_linear_strelPKhPKmmPmS3_Pl+0x80>
     798: cb08004a     	sub	x10, x2, x8
     79c: 8b080c28     	add	x8, x1, x8, lsl #3
     7a0: f840850b     	ldr	x11, [x8], #0x8
     7a4: 9b097d69     	mul	x9, x11, x9
     7a8: f100054a     	subs	x10, x10, #0x1
     7ac: 54ffffa1     	b.ne	0x7a0 <__Z15is_linear_strelPKhPKmmPmS3_Pl+0x70>
     7b0: f100013f     	cmp	x9, #0x0
     7b4: fa401844     	ccmp	x2, #0x0, #0x4, ne
     7b8: 54000440     	b.eq	0x840 <__Z15is_linear_strelPKhPKmmPmS3_Pl+0x110>
     7bc: d280000a     	mov	x10, #0x0               ; =0
     7c0: d2800008     	mov	x8, #0x0                ; =0
     7c4: d280000b     	mov	x11, #0x0               ; =0
     7c8: f86a782c     	ldr	x12, [x1, x10, lsl #3]
     7cc: f100059f     	cmp	x12, #0x1
```

## 3. Key Callees

