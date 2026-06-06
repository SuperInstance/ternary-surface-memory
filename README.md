# ternary-surface-memory

[![crate](https://img.shields.io/badge/crate-ternary--surface--memory-blue)](https://crates.io)
[![license](https://img.shields.io/badge/license-MIT%2FApache--2.0-green)](./LICENSE)

Surface memory for **ternary texture-like access** — a 2D memory layout with row/column addressing, bilinear interpolation (nearest-trit), region-to-region copy, and boundary handling modes (clamp, wrap, mirror).

## Overview

GPU surface memory provides 2D-addressable storage with hardware-accelerated boundary handling and interpolation. This crate brings that model to ternary (base-3) data, where each element is a **trit** (−1, 0, +1). It's designed for:

- **Ternary image processing** — filtering, resampling, and geometric transforms on trit-valued images
- **GPU kernel simulation** — modeling surface loads/stores before hardware execution
- **Texture-like access patterns** — 2D addressing with boundary modes for stencil computations

## Core Concepts

### Trit Values

Every element in a surface is a `Trit`: one of three states.

```rust
use ternary_surface_memory::Trit;

let t = Trit::PosOne;   // +1
let t = Trit::Zero;     //  0
let t = Trit::NegOne;   // -1

// Convert from/to integers
assert_eq!(Trit::from_i8(-1), Some(Trit::NegOne));
assert_eq!(Trit::PosOne.to_i8(), 1);

// Round floats to nearest trit
assert_eq!(Trit::nearest_trit(0.7), Trit::PosOne);
assert_eq!(Trit::nearest_trit(-0.4), Trit::Zero);
```

### Surface Memory

A 2D grid of trits with row/column addressing:

```rust
use ternary_surface_memory::*;

let mut surface = SurfaceMemory::new(64, 64); // 64×64 surface of Trit::Zero

// Write values
surface.set(SurfaceAddress::new(10, 20), Trit::PosOne);
surface.set(SurfaceAddress::new(11, 21), Trit::NegOne);

// Read values
assert_eq!(surface.get(SurfaceAddress::new(10, 20)), Some(Trit::PosOne));
```

### Boundary Modes

Out-of-bounds access is handled by three modes:

```rust
use ternary_surface_memory::*;

let mut surface = SurfaceMemory::new(4, 4);
surface.set(SurfaceAddress::new(0, 0), Trit::PosOne);
surface.set(SurfaceAddress::new(3, 3), Trit::NegOne);

// Clamp: coordinates clamped to [0, size-1]
assert_eq!(surface.get_with_boundary(-1, -1, BoundaryMode::Clamp), Trit::PosOne);
assert_eq!(surface.get_with_boundary(99, 99, BoundaryMode::Clamp), Trit::NegOne);

// Wrap: coordinates wrap around (modular)
assert_eq!(surface.get_with_boundary(0, 4, BoundaryMode::Wrap), Trit::PosOne); // col 4 → 0
assert_eq!(surface.get_with_boundary(-1, -1, BoundaryMode::Wrap), Trit::NegOne); // → (3,3)

// Mirror: coordinates bounce at boundaries
assert_eq!(surface.get_with_boundary(4, 4, BoundaryMode::Mirror), Trit::NegOne); // → (3,3)
```

### Bilinear Interpolation (Nearest-Trit)

Standard bilinear interpolation using four neighboring trits, rounded to the nearest valid trit value:

```rust
use ternary_surface_memory::*;

let mut surface = SurfaceMemory::new(4, 4);
surface.set(SurfaceAddress::new(0, 0), Trit::PosOne);
surface.set(SurfaceAddress::new(0, 1), Trit::NegOne);
surface.set(SurfaceAddress::new(1, 0), Trit::NegOne);
surface.set(SurfaceAddress::new(1, 1), Trit::PosOne);

// At exact grid point (0,0) → PosOne
assert_eq!(surface.bilinear_interpolate(0.0, 0.0, BoundaryMode::Clamp), Trit::PosOne);

// At (0.5, 0.5) → average of all four = 0.0 → Zero
assert_eq!(surface.bilinear_interpolate(0.5, 0.5, BoundaryMode::Clamp), Trit::Zero);
```

### Region Copy

Copy rectangular regions between surfaces (or within the same surface):

```rust
use ternary_surface_memory::*;

let src = SurfaceMemory::from_data(4, 4, vec![
    Trit::PosOne, Trit::NegOne, Trit::Zero, Trit::Zero,
    Trit::Zero,   Trit::PosOne, Trit::Zero, Trit::Zero,
    Trit::Zero,   Trit::Zero,   Trit::Zero, Trit::Zero,
    Trit::Zero,   Trit::Zero,   Trit::Zero, Trit::Zero,
]);

let mut dst = SurfaceMemory::new(4, 4);

// Copy 2×2 region from src(0,0) to dst(2,2)
SurfaceMemory::copy_region(
    &src,
    SurfaceAddress::new(0, 0),
    &mut dst,
    SurfaceAddress::new(2, 2),
    2,
    2,
);

assert_eq!(dst.get(SurfaceAddress::new(2, 2)), Some(Trit::PosOne));
assert_eq!(dst.get(SurfaceAddress::new(3, 3)), Some(Trit::PosOne));
```

## Key Types

| Type | Description |
|------|-------------|
| `Trit` | Ternary digit: NegOne (-1), Zero (0), PosOne (+1) |
| `SurfaceMemory` | 2D trit grid with addressing, interpolation, and boundaries |
| `SurfaceAddress` | (row, col) coordinate with linear address conversion |
| `BoundaryMode` | Clamp, Wrap, or Mirror handling for out-of-bounds access |

## Testing

```bash
cargo test
```

16 tests covering addressing, interpolation, surface copy, all three boundary modes, trit conversion, fill, and display formatting.

## License

MIT OR Apache-2.0
