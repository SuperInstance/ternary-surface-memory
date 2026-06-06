# ternary-surface-memory

2D surface memory for ternary data — texture-like addressing, bilinear interpolation (nearest-trit), region-to-region copy, and boundary handling (clamp, wrap, mirror). CPU-side simulation for ternary image processing and GPU kernel design.

## Why This Exists

GPU surface memory gives you 2D-addressable storage with hardware-accelerated boundary handling and interpolation. It's how GPUs do texture sampling: load a 2D grid, sample at fractional coordinates, handle edges gracefully. That model is perfect for ternary image processing — filtering, resampling, geometric transforms on trit-valued images.

But GPU surface memory requires, well, a GPU. And debugging 2D indexing bugs inside a kernel is an exercise in frustration. This crate brings the surface memory model to CPU: create a 2D grid of trits, read/write by `(row, col)`, sample at fractional coordinates with bilinear interpolation, handle out-of-bounds with clamp/wrap/mirror. Design your algorithm here, then port the verified logic to GPU.

## The Key Insight

Ternary bilinear interpolation has a built-in regularization effect. Standard bilinear interpolation produces arbitrary floating-point values. Ternary bilinear interpolation rounds to the nearest trit `{-1, 0, +1}` — so intermediate values snap to one of three states. This means interpolation can never produce an out-of-distribution value. A ternary image, no matter how much you resize or rotate it, stays ternary. That's a property binary and continuous representations can't match without explicit quantization steps.

## Quick Start

```rust
use ternary_surface_memory::*;

let mut surface = SurfaceMemory::new(64, 64); // 64×64 grid of Trit::Zero

// Write values
surface.set(SurfaceAddress::new(10, 20), Trit::PosOne);
surface.set(SurfaceAddress::new(11, 21), Trit::NegOne);

// Read values
assert_eq!(surface.get(SurfaceAddress::new(10, 20)), Some(Trit::PosOne));

// Boundary-handled access
surface.get_with_boundary(-1, -1, BoundaryMode::Clamp);  // → (0, 0) value
surface.get_with_boundary(99, 99, BoundaryMode::Wrap);   // → wraps around

// Bilinear interpolation at fractional coordinates
let interp = surface.bilinear_interpolate(10.5, 20.5, BoundaryMode::Clamp);
```

## Architecture

```
SurfaceMemory (width × height grid of Trit)
    │
    ├── Addressing
    │   get(addr) / set(addr, value)
    │   get_with_boundary(row, col, mode)
    │
    ├── Boundary Modes
    │   Clamp:  coord → [0, size-1]
    │   Wrap:   coord → coord % size
    │   Mirror: coord → bounce at edges
    │
    ├── Interpolation
    │   bilinear_interpolate(row, col, mode)
    │   → sample 4 neighbors → weighted avg → nearest trit
    │
    └── Region Operations
        copy_region(src, src_origin, dst, dst_origin, w, h)
```

## API Reference

### `Trit`

```rust
let t = Trit::NegOne;                     // -1
let t = Trit::Zero;                       //  0
let t = Trit::PosOne;                     // +1

Trit::from_i8(-1);                        // → Some(Trit::NegOne)
Trit::from_i8(2);                         // → None (invalid)
t.to_i8();                                // i8
t.to_f64();                               // f64
Trit::nearest_trit(0.7);                  // PosOne (rounds to nearest valid trit)
Trit::nearest_trit(-0.4);                 // Zero
```

### `SurfaceMemory`

```rust
let mut s = SurfaceMemory::new(width, height);
let mut s = SurfaceMemory::from_data(width, height, vec![Trit::...; w*h]);

s.get(SurfaceAddress::new(row, col));                           // Option<Trit>
s.set(SurfaceAddress::new(row, col), Trit::PosOne);             // bool (success)
s.get_with_boundary(row, col, mode);                            // Trit (always returns)
s.bilinear_interpolate(row_f64, col_f64, mode);                 // Trit
s.fill(Trit::Zero);                                             // fill entire surface
s.width() / s.height() / s.stride();
s.data() / s.data_mut();
s.len() / s.is_empty();
```

### `SurfaceAddress`

```rust
let addr = SurfaceAddress::new(row, col);
addr.to_linear(stride);   // u64 — flat index for row-major layout
```

### `BoundaryMode`

```rust
pub enum BoundaryMode {
    Clamp,    // coordinates clamped to [0, size-1]
    Wrap,     // modular wrap-around
    Mirror,   // bounce at boundaries
}
```

### Region Copy

```rust
SurfaceMemory::copy_region(
    &src, SurfaceAddress::new(src_row, src_col),
    &mut dst, SurfaceAddress::new(dst_row, dst_col),
    width, height,
) -> bool;  // false if out of bounds
```

## Real-World Example: Ternary Image Resizing

```rust
use ternary_surface_memory::*;

let src = SurfaceMemory::from_data(4, 4, vec![
    Trit::PosOne, Trit::PosOne, Trit::NegOne, Trit::NegOne,
    Trit::PosOne, Trit::PosOne, Trit::NegOne, Trit::NegOne,
    Trit::NegOne, Trit::NegOne, Trit::PosOne, Trit::PosOne,
    Trit::NegOne, Trit::NegOne, Trit::PosOne, Trit::PosOne,
]);

// Resize to 8×8 using bilinear interpolation
let mut dst = SurfaceMemory::new(8, 8);
let scale_x = 4.0 / 8.0;
let scale_y = 4.0 / 8.0;

for row in 0..8 {
    for col in 0..8 {
        let src_row = row as f64 * scale_y;
        let src_col = col as f64 * scale_x;
        let trit = src.bilinear_interpolate(src_row, src_col, BoundaryMode::Clamp);
        dst.set(SurfaceAddress::new(row, col), trit);
    }
}
// dst is now 8×8, but every pixel is still a valid trit {-1, 0, +1}
```

## Boundary Modes Explained

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Clamp** | Coordinates pinned to `[0, size-1]` | Edge-aware filters, CNN padding |
| **Wrap** | Modular: `coord % size` | Periodic patterns, tiled textures |
| **Mirror** | Bounce at edges: `0 1 2 1 0` | Symmetric boundary conditions, DCT-friendly |

Negative coordinates are handled in all modes. `Clamp(-1) = 0`, `Wrap(-1) = size-1`, `Mirror(-1)` bounces back.

## Ecosystem Connections

| Crate | Relationship |
|-------|-------------|
| `ternary-shared-memory` | 2D tiles loaded from surfaces into shared memory |
| `ternary-constant-cache` | Read-only surface data cached in constant cache |
| `ternary-warp-block` | Warp-cooperative surface loads/stores |
| `ternary-grid-launch` | 2D grid configs that map naturally to surface dimensions |

## Performance Characteristics

- **Point access**: O(1) — direct array index via `(row * stride + col)`
- **Bilinear interpolation**: O(1) — four point accesses + arithmetic
- **Region copy**: O(W × H) — one pass over source and destination
- **Boundary handling**: O(1) per coordinate — arithmetic only, no lookup tables
- **Memory**: O(W × H) — one `Trit` per pixel (1 byte per trit in current layout)

## Open Questions

- **Packed storage**: Current layout uses one byte per trit. Packed storage (2 bits per trit, 4 trits per byte) would reduce memory by 4× at the cost of bit manipulation on access.
- **Mipmaps**: Multi-resolution surface hierarchies for trilinear interpolation and level-of-detail selection.
- **GPU layout compatibility**: Match the exact memory layout of CUDA surface objects for seamless CPU↔GPU interop.

---

*16 tests · MIT OR Apache-2.0 · Zero dependencies*
