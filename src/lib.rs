//! # ternary-surface-memory
//!
//! Surface memory for ternary texture-like access.
//!
//! Provides 2D surface memory with row/column addressing, bilinear interpolation
//! (nearest-trit), region-to-region copy, and boundary handling (clamp, wrap, mirror).
//! Designed for GPU ternary kernel simulation and ternary image processing.

use std::fmt;

/// A single ternary digit (trit): -1, 0, or +1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Trit {
    NegOne = -1,
    Zero = 0,
    PosOne = 1,
}

impl Trit {
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Trit::NegOne),
            0 => Some(Trit::Zero),
            1 => Some(Trit::PosOne),
            _ => None,
        }
    }

    pub fn to_i8(self) -> i8 {
        self as i8
    }

    pub fn to_f64(self) -> f64 {
        self as i8 as f64
    }

    /// Round a float to the nearest trit value.
    pub fn nearest_trit(value: f64) -> Self {
        let rounded = value.round() as i8;
        match rounded.clamp(-1, 1) {
            -1 => Trit::NegOne,
            0 => Trit::Zero,
            1 => Trit::PosOne,
            _ => Trit::Zero,
        }
    }
}

impl fmt::Display for Trit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_i8())
    }
}

/// Boundary mode for out-of-bounds access.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryMode {
    /// Clamp coordinates to valid range.
    Clamp,
    /// Wrap coordinates using modular arithmetic.
    Wrap,
    /// Mirror coordinates at boundaries.
    Mirror,
}

/// 2D surface address (row, column).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct SurfaceAddress {
    pub row: u32,
    pub col: u32,
}

impl SurfaceAddress {
    pub fn new(row: u32, col: u32) -> Self {
        Self { row, col }
    }

    /// Convert to a linear index for a surface with the given stride.
    pub fn to_linear(&self, stride: u32) -> u64 {
        self.row as u64 * stride as u64 + self.col as u64
    }
}

/// A 2D surface of ternary values.
#[derive(Debug, Clone)]
pub struct SurfaceMemory {
    width: u32,
    height: u32,
    stride: u32,
    data: Vec<Trit>,
}

impl SurfaceMemory {
    /// Create a new surface filled with Trit::Zero.
    pub fn new(width: u32, height: u32) -> Self {
        let stride = width;
        let data = vec![Trit::Zero; (width * height) as usize];
        Self {
            width,
            height,
            stride,
            data,
        }
    }

    /// Create a surface from raw trit data.
    pub fn from_data(width: u32, height: u32, data: Vec<Trit>) -> Self {
        assert_eq!(data.len(), (width * height) as usize);
        Self {
            width,
            height,
            stride: width,
            data,
        }
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    pub fn stride(&self) -> u32 {
        self.stride
    }

    /// Get a trit at the given address. Returns None if out of bounds.
    pub fn get(&self, addr: SurfaceAddress) -> Option<Trit> {
        if addr.row < self.height && addr.col < self.width {
            let idx = addr.to_linear(self.stride) as usize;
            self.data.get(idx).copied()
        } else {
            None
        }
    }

    /// Set a trit at the given address. Returns false if out of bounds.
    pub fn set(&mut self, addr: SurfaceAddress, value: Trit) -> bool {
        if addr.row < self.height && addr.col < self.width {
            let idx = addr.to_linear(self.stride) as usize;
            self.data[idx] = value;
            true
        } else {
            false
        }
    }

    /// Get a trit with boundary handling.
    pub fn get_with_boundary(&self, row: i64, col: i64, mode: BoundaryMode) -> Trit {
        let r = self.apply_boundary(row, self.height, mode);
        let c = self.apply_boundary(col, self.width, mode);
        self.data[(r * self.stride as u64 + c) as usize]
    }

    /// Apply boundary mode to a coordinate.
    fn apply_boundary(&self, coord: i64, size: u32, mode: BoundaryMode) -> u64 {
        let s = size as i64;
        match mode {
            BoundaryMode::Clamp => coord.clamp(0, s - 1) as u64,
            BoundaryMode::Wrap => {
                let result = coord % s;
                if result < 0 { (result + s) as u64 } else { result as u64 }
            }
            BoundaryMode::Mirror => {
                if coord < 0 {
                    let abs = (-coord) as u64;
                    let idx = abs % (size as u64);
                    // Mirror bounce
                    if (abs / (size as u64)) % 2 == 0 {
                        idx.min(size as u64 - 1)
                    } else {
                        (size as u64 - 1).saturating_sub(idx)
                    }
                } else if coord >= s {
                    let overshoot = (coord - s) as u64;
                    let total = size as u64;
                    let idx = overshoot % total;
                    if (overshoot / total) % 2 == 0 {
                        (total - 1).saturating_sub(idx.min(total - 1))
                    } else {
                        idx.min(total - 1)
                    }
                } else {
                    coord as u64
                }
            }
        }
    }

    /// Bilinear interpolation of ternary values, rounded to nearest trit.
    /// Uses the four nearest grid points and returns the nearest-trit result.
    pub fn bilinear_interpolate(
        &self,
        row: f64,
        col: f64,
        mode: BoundaryMode,
    ) -> Trit {
        let r0 = row.floor() as i64;
        let r1 = r0 + 1;
        let c0 = col.floor() as i64;
        let c1 = c0 + 1;

        let tr0c0 = self.get_with_boundary(r0, c0, mode).to_f64();
        let tr0c1 = self.get_with_boundary(r0, c1, mode).to_f64();
        let tr1c0 = self.get_with_boundary(r1, c0, mode).to_f64();
        let tr1c1 = self.get_with_boundary(r1, c1, mode).to_f64();

        let dr = row - r0 as f64;
        let dc = col - c0 as f64;

        let top = tr0c0 * (1.0 - dc) + tr0c1 * dc;
        let bottom = tr1c0 * (1.0 - dc) + tr1c1 * dc;
        let value = top * (1.0 - dr) + bottom * dr;

        Trit::nearest_trit(value)
    }

    /// Copy a rectangular region from one surface to another (or within the same surface).
    pub fn copy_region(
        src: &SurfaceMemory,
        src_origin: SurfaceAddress,
        dst: &mut SurfaceMemory,
        dst_origin: SurfaceAddress,
        width: u32,
        height: u32,
    ) -> bool {
        // Bounds check source
        if src_origin.row + height > src.height || src_origin.col + width > src.width {
            return false;
        }
        // Bounds check destination
        if dst_origin.row + height > dst.height || dst_origin.col + width > dst.width {
            return false;
        }

        // Copy row by row (handles overlapping correctly if src == dst with different origins)
        for dr in 0..height {
            let src_row = src_origin.row + dr;
            let dst_row = dst_origin.row + dr;
            for dc in 0..width {
                let src_col = src_origin.col + dc;
                let dst_col = dst_origin.col + dc;
                let val = src.data[(src_row as u64 * src.stride as u64 + src_col as u64) as usize];
                dst.data[(dst_row as u64 * dst.stride as u64 + dst_col as u64) as usize] = val;
            }
        }
        true
    }

    /// Fill the entire surface with a trit value.
    pub fn fill(&mut self, value: Trit) {
        for t in &mut self.data {
            *t = value;
        }
    }

    /// Get the raw data slice.
    pub fn data(&self) -> &[Trit] {
        &self.data
    }

    /// Get the raw data slice mutably.
    pub fn data_mut(&mut self) -> &mut Vec<Trit> {
        &mut self.data
    }

    /// Total number of trits in the surface.
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Whether the surface is empty.
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_addressing_correctness() {
        let mut surface = SurfaceMemory::new(4, 3); // 4 wide, 3 tall
        surface.set(SurfaceAddress::new(0, 0), Trit::PosOne);
        surface.set(SurfaceAddress::new(1, 2), Trit::NegOne);
        surface.set(SurfaceAddress::new(2, 3), Trit::PosOne);

        assert_eq!(surface.get(SurfaceAddress::new(0, 0)), Some(Trit::PosOne));
        assert_eq!(surface.get(SurfaceAddress::new(1, 2)), Some(Trit::NegOne));
        assert_eq!(surface.get(SurfaceAddress::new(2, 3)), Some(Trit::PosOne));
        assert_eq!(surface.get(SurfaceAddress::new(0, 1)), Some(Trit::Zero));

        // Out of bounds
        assert_eq!(surface.get(SurfaceAddress::new(3, 0)), None);
        assert_eq!(surface.get(SurfaceAddress::new(0, 4)), None);
    }

    #[test]
    fn test_linear_address() {
        let addr = SurfaceAddress::new(2, 3);
        assert_eq!(addr.to_linear(5), 13);
    }

    #[test]
    fn test_interpolation_nearest_trit() {
        // Create a small surface
        let mut surface = SurfaceMemory::new(4, 4);
        surface.set(SurfaceAddress::new(0, 0), Trit::PosOne);
        surface.set(SurfaceAddress::new(0, 1), Trit::PosOne);
        surface.set(SurfaceAddress::new(1, 0), Trit::NegOne);
        surface.set(SurfaceAddress::new(1, 1), Trit::NegOne);

        // At (0, 0) → should return PosOne (exact grid point)
        assert_eq!(
            surface.bilinear_interpolate(0.0, 0.0, BoundaryMode::Clamp),
            Trit::PosOne
        );

        // At (1, 1) → should return NegOne (exact grid point)
        assert_eq!(
            surface.bilinear_interpolate(1.0, 1.0, BoundaryMode::Clamp),
            Trit::NegOne
        );

        // At (0.5, 0.0) → interpolation of PosOne and NegOne at 50% = 0.0 → Zero
        assert_eq!(
            surface.bilinear_interpolate(0.5, 0.0, BoundaryMode::Clamp),
            Trit::Zero
        );
    }

    #[test]
    fn test_surface_copy_preserves_data() {
        let mut src = SurfaceMemory::new(4, 4);
        src.set(SurfaceAddress::new(0, 0), Trit::PosOne);
        src.set(SurfaceAddress::new(0, 1), Trit::NegOne);
        src.set(SurfaceAddress::new(1, 0), Trit::Zero);
        src.set(SurfaceAddress::new(1, 1), Trit::PosOne);

        let mut dst = SurfaceMemory::new(4, 4);
        let ok = SurfaceMemory::copy_region(
            &src,
            SurfaceAddress::new(0, 0),
            &mut dst,
            SurfaceAddress::new(2, 2),
            2,
            2,
        );
        assert!(ok);
        assert_eq!(dst.get(SurfaceAddress::new(2, 2)), Some(Trit::PosOne));
        assert_eq!(dst.get(SurfaceAddress::new(2, 3)), Some(Trit::NegOne));
        assert_eq!(dst.get(SurfaceAddress::new(3, 2)), Some(Trit::Zero));
        assert_eq!(dst.get(SurfaceAddress::new(3, 3)), Some(Trit::PosOne));
    }

    #[test]
    fn test_surface_copy_out_of_bounds() {
        let src = SurfaceMemory::new(2, 2);
        let mut dst = SurfaceMemory::new(2, 2);
        // Source region extends beyond source
        assert!(!SurfaceMemory::copy_region(
            &src,
            SurfaceAddress::new(0, 0),
            &mut dst,
            SurfaceAddress::new(0, 0),
            3,
            3,
        ));
    }

    #[test]
    fn test_clamp_boundary() {
        let mut surface = SurfaceMemory::new(3, 3);
        surface.set(SurfaceAddress::new(0, 0), Trit::PosOne);
        surface.set(SurfaceAddress::new(2, 2), Trit::NegOne);

        // Clamp: negative coords clamp to 0
        assert_eq!(surface.get_with_boundary(-1, -1, BoundaryMode::Clamp), Trit::PosOne);
        // Clamp: beyond edge clamps to last
        assert_eq!(surface.get_with_boundary(5, 5, BoundaryMode::Clamp), Trit::NegOne);
        assert_eq!(surface.get_with_boundary(2, 5, BoundaryMode::Clamp), Trit::NegOne);
    }

    #[test]
    fn test_wrap_boundary() {
        let mut surface = SurfaceMemory::new(4, 4);
        surface.set(SurfaceAddress::new(0, 0), Trit::PosOne);
        surface.set(SurfaceAddress::new(3, 3), Trit::NegOne);

        // Wrap: col 4 wraps to 0, row 4 wraps to 0
        assert_eq!(surface.get_with_boundary(0, 4, BoundaryMode::Wrap), Trit::PosOne);
        assert_eq!(surface.get_with_boundary(4, 0, BoundaryMode::Wrap), Trit::PosOne);

        // Wrap: negative wraps around: -1 % 4 = 3
        assert_eq!(surface.get_with_boundary(-1, -1, BoundaryMode::Wrap), Trit::NegOne);
    }

    #[test]
    fn test_mirror_boundary() {
        let mut surface = SurfaceMemory::new(4, 4);
        surface.set(SurfaceAddress::new(3, 3), Trit::PosOne);
        surface.set(SurfaceAddress::new(0, 0), Trit::NegOne);

        // coord 4 in size 4: overshoot by 1, mirror → 3
        assert_eq!(surface.get_with_boundary(4, 4, BoundaryMode::Mirror), Trit::PosOne);

        // coord -1 in size 4: abs = 1, 1 % 4 = 1, bounce 0 → index 1
        // row=-1 col=-1 → row=1 col=1 → which is Zero (default)
        assert_eq!(surface.get_with_boundary(-1, 0, BoundaryMode::Mirror), Trit::Zero);

        // coord 0,0 is NegOne
        assert_eq!(surface.get_with_boundary(0, 0, BoundaryMode::Mirror), Trit::NegOne);
    }

    #[test]
    fn test_trit_from_i8() {
        assert_eq!(Trit::from_i8(-1), Some(Trit::NegOne));
        assert_eq!(Trit::from_i8(0), Some(Trit::Zero));
        assert_eq!(Trit::from_i8(1), Some(Trit::PosOne));
        assert_eq!(Trit::from_i8(2), None);
        assert_eq!(Trit::from_i8(-2), None);
    }

    #[test]
    fn test_trit_nearest() {
        assert_eq!(Trit::nearest_trit(0.7), Trit::PosOne);
        assert_eq!(Trit::nearest_trit(0.4), Trit::Zero);
        assert_eq!(Trit::nearest_trit(-0.6), Trit::NegOne);
        assert_eq!(Trit::nearest_trit(1.5), Trit::PosOne);
        assert_eq!(Trit::nearest_trit(-1.5), Trit::NegOne);
    }

    #[test]
    fn test_fill() {
        let mut surface = SurfaceMemory::new(3, 3);
        surface.fill(Trit::PosOne);
        for row in 0..3 {
            for col in 0..3 {
                assert_eq!(surface.get(SurfaceAddress::new(row, col)), Some(Trit::PosOne));
            }
        }
    }

    #[test]
    fn test_surface_from_data() {
        let data = vec![
            Trit::PosOne, Trit::NegOne, Trit::Zero,
            Trit::Zero, Trit::PosOne, Trit::NegOne,
        ];
        let surface = SurfaceMemory::from_data(3, 2, data);
        assert_eq!(surface.get(SurfaceAddress::new(0, 0)), Some(Trit::PosOne));
        assert_eq!(surface.get(SurfaceAddress::new(1, 2)), Some(Trit::NegOne));
        assert_eq!(surface.len(), 6);
    }

    #[test]
    fn test_interpolation_with_boundary() {
        let mut surface = SurfaceMemory::new(2, 2);
        surface.set(SurfaceAddress::new(0, 0), Trit::PosOne);
        surface.set(SurfaceAddress::new(0, 1), Trit::NegOne);
        surface.set(SurfaceAddress::new(1, 0), Trit::NegOne);
        surface.set(SurfaceAddress::new(1, 1), Trit::PosOne);

        // Interpolate at (0.5, 0.5): average of all four = 0.0 → Zero
        assert_eq!(
            surface.bilinear_interpolate(0.5, 0.5, BoundaryMode::Clamp),
            Trit::Zero
        );
    }

    #[test]
    fn test_set_out_of_bounds_returns_false() {
        let mut surface = SurfaceMemory::new(2, 2);
        assert!(!surface.set(SurfaceAddress::new(2, 0), Trit::PosOne));
        assert!(!surface.set(SurfaceAddress::new(0, 2), Trit::PosOne));
    }

    #[test]
    fn test_surface_copy_self() {
        let surface = SurfaceMemory::from_data(4, 4, vec![
            Trit::PosOne, Trit::NegOne, Trit::Zero, Trit::Zero,
            Trit::Zero, Trit::Zero, Trit::Zero, Trit::Zero,
            Trit::Zero, Trit::Zero, Trit::Zero, Trit::Zero,
            Trit::Zero, Trit::Zero, Trit::Zero, Trit::Zero,
        ]);
        let mut dst = surface.clone();

        // Copy within same surface: (0,0)-(1,1) → (2,2)-(3,3)
        let ok = SurfaceMemory::copy_region(
            &surface,
            SurfaceAddress::new(0, 0),
            &mut dst,
            SurfaceAddress::new(2, 2),
            2,
            2,
        );
        assert!(ok);
        assert_eq!(dst.get(SurfaceAddress::new(2, 2)), Some(Trit::PosOne));
        assert_eq!(dst.get(SurfaceAddress::new(2, 3)), Some(Trit::NegOne));
    }

    #[test]
    fn test_display_trit() {
        assert_eq!(format!("{}", Trit::NegOne), "-1");
        assert_eq!(format!("{}", Trit::Zero), "0");
        assert_eq!(format!("{}", Trit::PosOne), "1");
    }
}
