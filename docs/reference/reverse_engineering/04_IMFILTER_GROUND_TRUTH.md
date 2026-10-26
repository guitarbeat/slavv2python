# Ground-Truth Architecture Specification: N-D Spatial Convolution

- **Subsystem:** N-D Spatial Convolution & Boundary Padding
- **Native Binary:** `/Volumes/LoveSSD/Applications/MATLAB_R2024b.app/bin/maca64/libmwimfilter.dylib`
- **Primary Function:** `imfilter_real64`
- **Symbol:** `_imfilter_real64`
- **Instructions Decoded:** 1100 | **Branches:** 186 | **Calls:** 71
- **Corresponding Python Implementation:** [`slavv_python.pipeline.energy.filter`](file:///Volumes/LoveSSD/NeuroVASC/repos/slavv2python/slavv_python/pipeline/energy/filter.py)

---

## 1. Algorithmic Overview

1. **Precision & Accumulation:**
   - `imfilter_real64`: Accumulates in 64-bit IEEE double registers (`d8`-`d11`).
   - Prevents catastrophic cancellation in multi-scale Hessian second-order differential filtering.
2. **Boundary Extension (`update_linear_index`):**
   - Supports 4 boundary conditions:
     - `replicate`: Clamps out-of-bound indices to boundary voxels: min(max(i, 0), N-1).
     - `symmetric`: Mirrors indices across the boundary edge.
     - `circular`: Wraps indices modulo grid size: i mod N.
     - `constant`: Injects constant fill (typically 0.0).
3. **Correlation vs Convolution:**
   - MATLAB `imfilter` performs **multidimensional correlation** by default (kernel is NOT reversed).
   - Python `scipy.ndimage.convolve` reverses the kernel, requiring `scipy.ndimage.correlate` for exact parity.

## 2. Disassembly Excerpt (`_imfilter_real64`)

```arm64
    a6cc: d105c3ff     	sub	sp, sp, #0x170
    a6d0: 6d0f2beb     	stp	d11, d10, [sp, #0xf0]
    a6d4: 6d1023e9     	stp	d9, d8, [sp, #0x100]
    a6d8: a9116ffc     	stp	x28, x27, [sp, #0x110]
    a6dc: a91267fa     	stp	x26, x25, [sp, #0x120]
    a6e0: a9135ff8     	stp	x24, x23, [sp, #0x130]
    a6e4: a91457f6     	stp	x22, x21, [sp, #0x140]
    a6e8: a9154ff4     	stp	x20, x19, [sp, #0x150]
    a6ec: a9167bfd     	stp	x29, x30, [sp, #0x160]
    a6f0: 910583fd     	add	x29, sp, #0x160
    a6f4: a9060fe7     	stp	x7, x3, [sp, #0x60]
    a6f8: f9003be6     	str	x6, [sp, #0x70]
    a6fc: a90413e5     	stp	x5, x4, [sp, #0x40]
    a700: 1e604089     	fmov	d9, d4
    a704: 1e60406b     	fmov	d11, d3
    a708: 1e604048     	fmov	d8, d2
    a70c: 1e60402a     	fmov	d10, d1
    a710: a90307e0     	stp	x0, x1, [sp, #0x30]
    a714: 394047a8     	ldrb	w8, [x29, #0x11]
    a718: b9002fe8     	str	w8, [sp, #0x2c]
    a71c: 394043a8     	ldrb	w8, [x29, #0x10]
    a720: b9002be8     	str	w8, [sp, #0x28]
    a724: a9377fbf     	stp	xzr, xzr, [x29, #-0x90]
    a728: f81803bf     	stur	xzr, [x29, #-0x80]
    a72c: a935ffbf     	stp	xzr, xzr, [x29, #-0xa8]
    a730: f81683bf     	stur	xzr, [x29, #-0x98]
    a734: a90a7fff     	stp	xzr, xzr, [sp, #0xa0]
    a738: f9005bff     	str	xzr, [sp, #0xb0]
    a73c: a908ffff     	stp	xzr, xzr, [sp, #0x88]
    a740: f9004fff     	str	xzr, [sp, #0x98]
    a744: 9e79001a     	fcvtzu	x26, d0
    a748: f9002ffa     	str	x26, [sp, #0x58]
    a74c: b4000d3a     	cbz	x26, 0xa8f0 <_imfilter_real64+0x224>
    a750: aa0203f9     	mov	x25, x2
    a754: d2800003     	mov	x3, #0x0                ; =0
    a758: d280001c     	mov	x28, #0x0               ; =0
    a75c: d2800018     	mov	x24, #0x0               ; =0
    a760: d2800015     	mov	x21, #0x0               ; =0
    a764: f85703b7     	ldur	x23, [x29, #-0x90]
    a768: 14000008     	b	0xa788 <_imfilter_real64+0xbc>
```

## 3. Key Callees
- `dyld_stub_binder+0xb868`
- `dyld_stub_binder+0xb85c`
- `_imfilter_real64+0x978`
- `_imfilter_real64+0xd88`
- `_imfilter_real64+0xe10`
- `dyld_stub_binder+0xb7fc`
- `dyld_stub_binder+0xb838`
- `dyld_stub_binder+0xb82c`
- `_imfilter_real64+0xf44`
- `dyld_stub_binder+0xb814`
- `dyld_stub_binder+0xb8a4`
- `dyld_stub_binder+0xb8b0`
- `dyld_stub_binder+0xb808`
- `dyld_stub_binder+0xb898`
- `dyld_stub_binder+0xb820`
- `_imfilter_real64+0xf30`
- `_imfilter_real64+0xe38`
- `_imfilter_real64+0xeac`
- `_imfilter_real64+0xd9c`
- `dyld_stub_binder+0xb874`
- `_imfilter_real64+0xdec`
- `dyld_stub_binder+0xb88c`
- `dyld_stub_binder+0xb880`
- `dyld_stub_binder+0xb844`
- `dyld_stub_binder+0xb850`
