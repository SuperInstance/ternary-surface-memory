# ternary-surface-memory

2D surface memory for ternary texture-like access.

GPU surface memory provides 2D-addressable storage with hardware-accelerated boundary handling and interpolation. This crate brings that model to ternary (base-3) data, where every element is a trit ∈ {−1, 0, +1}. You get row/column addressing, three boundary modes (clamp, wrap, mirror), bilinear interpolation rounded to the nearest trit, and region-to-region copy — all designed for ternary image processing and GPU kernel simulation.

The key insight: ternary images have unique interpolation semantics. When you bilinearly interpolate between −1 and +1 at the midpoint, you get 0 — a valid trit. But interpolating between two −1 values gives −1 (not −2), because we round to the nearest valid trit. This "nearest-trit" interpolation preserves the ternary alphabet without introducing fractional values.

## Why This Exists

Ternary neural networks work with quantized activations and weights in {−1, 0, +1}. When you need to:
- **Resize** a ternary feature map (e.g., upsample in a U-Net decoder)
- **Apply geometric transforms** (rotation, translation) to ternary images
- **Implement stencil operations** (convolution, pooling) with boundary handling
- **Copy tiles** between surfaces (loading data into shared memory tiles)

...you need 2D memory with boundary modes and interpolation. This crate provides exactly that, with trit-native semantics.

## Quick Start

### Create and Access

```rust
use ternary_surface_memory::*;

// Create a 64×64 surface filled with Trit::Zero
let mut surface = SurfaceMemory::new(64, 64);

// Write values
surface.set(SurfaceAddress::new(10, 20), Trit::PosOne);
surface.set(SurfaceAddress::new(11, 21), Trit::NegOne);

// Read values
assert_eq!(surface.get(SurfaceAddress::new(10, 20)), Some(Trit::PosOne));

// Out-of-bounds returns None
assert_eq!(surface.get(SurfaceAddress::new(100, 0)), None);
```

### Boundary Modes

```rust
use ternary_surface_memory::*;

let mut surface = SurfaceMemory::new(4, 4);
surface.set(SurfaceAddress::new(0, 0), Trit::PosOne);
surface.set(SurfaceAddress::new(3, 3), Trit::NegOne);

// Clamp: coordinates clamped to valid range
assert_eq!(surface.get_with_boundary(-1, -1, BoundaryMode::Clamp), Trit::PosOne);
assert_eq!(surface.get_with_boundary(99, 99, BoundaryMode::Clamp), Trit::NegOne);

// Wrap: coordinates wrap around (modular)
assert_eq!(surface.get_with_boundary(0, 4, BoundaryMode::Wrap), Trit::PosOne);  // col 4 → 0
assert_eq!(surface.get_with_boundary(-1, -1, BoundaryMode::Wrap), Trit::NegOne); // → (3,3)

// Mirror: coordinates bounce at edges
assert_eq!(surface.get_with_boundary(4, 4, BoundaryMode::Mirror), Trit::NegOne); // → (3,3)
```

### Bilinear Interpolation

```rust
use ternary_surface_memory::*;

let mut surface = SurfaceMemory::new(4, 4);
surface.set(SurfaceAddress::new(0, 0), Trit::PosOne);   // +1
surface.set(SurfaceAddress::new(0, 1), Trit::NegOne);   // -1
surface.set(SurfaceAddress::new(1, 0), Trit::NegOne);   // -1
surface.set(SurfaceAddress::new(1, 1), Trit::PosOne);   // +1

// Exact grid point → returns that trit
assert_eq!(surface.bilinear_interpolate(0.0, 0.0, BoundaryMode::Clamp), Trit::PosOne);

// Center of 2×2 cell: average = (+1 + -1 + -1 + +1) / 4 = 0 → Zero
assert_eq!(surface.bilinear_interpolate(0.5, 0.5, BoundaryMode::Clamp), Trit::Zero);

// Halfway between row 0 and row 1 at col 0: (+1 + -1) / 2 = 0 → Zero
assert_eq!(surface.bilinear_interpolate(0.5, 0.0, BoundaryMode::Clamp), Trit::Zero);
```

### Region Copy

```rust
use ternary_surface_memory::*;

let src = SurfaceMemory::from_data(4, 4, vec![
    Trit::PosOne, Trit::NegOne, Trit::Zero, Trit::Zero,
    Trit::Zero,   Trit::PosOne, Trit::Zero, Trit::Zero,
    Trit::Zero,   Trit::Zero,   Trit::Zero, Trit::Zero,
    Trit::Zero,   Trit::Zero,   Trit::Zero, Trit::Zero,
]);

let mut dst = SurfaceMemory::new(4, 4);

// Copy 2×2 region from (0,0) to (2,2)
SurfaceMemory::copy_region(
    &src, SurfaceAddress::new(0, 0),
    &mut dst, SurfaceAddress::new(2, 2),
    2, 2,
);

assert_eq!(dst.get(SurfaceAddress::new(2, 2)), Some(Trit::PosOne));
assert_eq!(dst.get(SurfaceAddress::new(2, 3)), Some(Trit::NegOne));
```

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                   SurfaceMemory                           │
│                                                          │
│  Storage: Vec<Trit> (row-major)                          │
│  Dimensions: width × height, stride = width              │
│                                                          │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────┐ │
│  │  Addressing  │  │  Boundaries  │  │ Interpolation  │ │
│  │  get / set   │  │  Clamp       │  │ Bilinear       │ │
│  │  row/col     │  │  Wrap        │  │ nearest-trit   │ │
│  │  linear idx  │  │  Mirror      │  │ rounding       │ │
│  └─────────────┘  └──────────────┘  └────────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │  Region Copy: src(rect) → dst(rect)              │   │
│  └──────────────────────────────────────────────────┘   │
├──────────────────────────────────────────────────────────┤
│  Trit: NegOne(-1) | Zero(0) | PosOne(+1)                │
│  from_i8, to_i8, to_f64, nearest_trit, Display          │
├──────────────────────────────────────────────────────────┤
│  SurfaceAddress { row, col }                             │
│  to_linear(stride) → u64                                 │
└──────────────────────────────────────────────────────────┘
```

### Trit

The fundamental value type. Three states, no more, no less:

```rust
let t = Trit::from_i8(1).unwrap();  // PosOne
assert_eq!(t.to_i8(), 1);
assert_eq!(t.to_f64(), 1.0);
assert_eq!(format!("{}", t), "1");

// Round floats to nearest trit
assert_eq!(Trit::nearest_trit(0.7), Trit::PosOne);   // rounds to 1
assert_eq!(Trit::nearest_trit(0.4), Trit::Zero);     // rounds to 0
assert_eq!(Trit::nearest_trit(-0.6), Trit::NegOne);  // rounds to -1
assert_eq!(Trit::nearest_trit(1.5), Trit::PosOne);   // clamps to 1
assert_eq!(Trit::nearest_trit(-1.5), Trit::NegOne);  // clamps to -1
```

### Boundary Modes

Three strategies for out-of-bounds coordinates:

| Mode | Behavior | Use case |
|------|----------|----------|
| `Clamp` | Clamp to [0, size−1] | Object detection (don't wrap) |
| `Wrap` | Modular: coord % size | Periodic textures, toroidal topology |
| `Mirror` | Bounce at edges | Symmetric padding, reflection padding |

Clamp is the default for most computer vision tasks. Wrap is useful for periodic data. Mirror is the correct padding for symmetric boundary conditions in stencil operations.

### Bilinear Interpolation (Nearest-Trit)

Standard bilinear interpolation using four neighboring trits, then rounded to the nearest valid trit:

```
value = (1-dr)(1-dc)·T(r0,c0) + (1-dr)dc·T(r0,c1) + dr(1-dc)·T(r1,c0) + dr·dc·T(r1,c1)
result = nearest_trit(value)
```

The `nearest_trit` function rounds to the closest integer in {−1, 0, +1} and clamps. This means:
- Interpolating between same-valued trits returns that trit
- Interpolating between −1 and +1 at 50% gives 0 (exact midpoint)
- Interpolating near 0.5 rounds to 1, near −0.5 rounds to −1

## API Reference

### `SurfaceMemory`

| Method | Description |
|--------|-------------|
| `new(width, height)` | Create zeroed surface |
| `from_data(width, height, data)` | Create from existing trit data |
| `get(addr)` / `set(addr, value)` | In-bounds access |
| `get_with_boundary(row, col, mode)` | Access with boundary handling |
| `bilinear_interpolate(row, col, mode)` | Interpolated access |
| `fill(value)` | Fill entire surface |
| `data()` / `data_mut()` | Raw data access |
| `width()` / `height()` / `stride()` | Dimensions |
| `len()` / `is_empty()` | Size queries |

### `SurfaceMemory::copy_region` (static method)

```rust
SurfaceMemory::copy_region(
    src, src_origin,
    dst, dst_origin,
    width, height
) → bool  // false if out of bounds
```

Copies between different surfaces or within the same surface (handles overlapping correctly by copying row-by-row).

### `SurfaceAddress`

| Method | Description |
|--------|-------------|
| `new(row, col)` | Create address |
| `to_linear(stride)` | Convert to flat index |

### `Trit`

| Method | Description |
|--------|-------------|
| `from_i8(v)` | Convert (returns None for invalid) |
| `to_i8()` / `to_f64()` | Convert out |
| `nearest_trit(value)` | Round float to nearest trit |

### `BoundaryMode`

`Clamp` | `Wrap` | `Mirror`

## Real-World Example: Ternary Feature Map Upsampling

```rust
use ternary_surface_memory::*;

// Input: 4×4 ternary feature map from a neural network
let input = SurfaceMemory::from_data(4, 4, vec![
    Trit::NegOne, Trit::NegOne, Trit::PosOne, Trit::PosOne,
    Trit::NegOne, Trit::NegOne, Trit::PosOne, Trit::PosOne,
    Trit::PosOne, Trit::PosOne, Trit::NegOne, Trit::NegOne,
    Trit::PosOne, Trit::PosOne, Trit::NegOne, Trit::NegOne,
]);

// Upsample to 8×8 using bilinear interpolation
let mut output = SurfaceMemory::new(8, 8);

for row in 0..8 {
    for col in 0..8 {
        // Map output coordinates to input coordinates
        let src_row = row as f64 * 3.0 / 7.0;  // 0..3 range
        let src_col = col as f64 * 3.0 / 7.0;
        
        let interpolated = input.bilinear_interpolate(src_row, src_col, BoundaryMode::Clamp);
        output.set(SurfaceAddress::new(row, col), interpolated);
    }
}

// The output is a 2× upsampled version, preserving ternary values
// (corners remain the same, edges get nearest-trit interpolation)
assert_eq!(output.get(SurfaceAddress::new(0, 0)), Some(Trit::NegOne));
assert_eq!(output.get(SurfaceAddress::new(7, 7)), Some(Trit::NegOne));
```

## Real-World Example: Ternary Convolution with Boundary Padding

```rust
use ternary_surface_memory::*;

fn ternary_convolve(input: &SurfaceMemory, kernel: &[[Trit; 3]; 3]) -> SurfaceMemory {
    let (h, w) = (input.height(), input.width());
    let mut output = SurfaceMemory::new(w, h);
    
    for row in 0..h {
        for col in 0..w {
            let mut acc: f64 = 0.0;
            for kr in 0..3 {
                for kc in 0..3 {
                    let val = input.get_with_boundary(
                        row as i64 + kr as i64 - 1,
                        col as i64 + kc as i64 - 1,
                        BoundaryMode::Mirror, // symmetric padding
                    );
                    acc += val.to_f64() * kernel[kr][kc].to_f64();
                }
            }
            output.set(SurfaceAddress::new(row, col), Trit::nearest_trit(acc));
        }
    }
    output
}
```

## Ecosystem Connections

- **`ternary-shared-memory`** — Shared memory tiles that surface data is loaded into for GPU processing
- **`ternary-warp-block`** — Warp operations that read from surface-like 2D layouts
- **`ternary-grid-launch`** — Grid dimensions for kernels that process surface data
- **`ternary-constant-cache`** — Cache simulation for read-only surface data (e.g., weight matrices)

## Performance Notes

- **Access**: O(1) for `get`/`set` — direct indexing into a flat Vec.
- **Boundary handling**: O(1) — arithmetic on coordinates. Mirror mode is the most complex but still constant time.
- **Interpolation**: O(1) — reads 4 neighbors and computes weighted average.
- **Region copy**: O(width × height) for the copied region. Row-by-row copy handles overlapping regions correctly.
- **Memory**: width × height × sizeof(Trit) = width × height bytes. A 1024×1024 ternary image is just 1 MB.

## Open Questions

- **Mipmapping**: No support for multi-resolution surfaces. Would be useful for trilinear interpolation in ternary texture sampling.
- **Packed storage**: Currently one byte per trit. Could pack 4 trits per byte (2 bits each) for 4× memory reduction.
- **3D surfaces**: Only 2D. Volumetric ternary data would need a 3D extension.
- **Batch operations**: No vectorized fill or map operations. Would benefit from SIMD for bulk processing.

## License

MIT OR Apache-2.0
