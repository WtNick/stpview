//! Minimal linear algebra. Geometry is evaluated in f64 (CAD models routinely
//! span 1e3 mm with 1e-6 mm features); only the vertex buffer drops to f32.

use std::ops::{Add, AddAssign, Div, Mul, Neg, Sub};

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

pub const fn v3(x: f64, y: f64, z: f64) -> Vec3 {
    Vec3 { x, y, z }
}

impl Vec3 {
    pub const ZERO: Vec3 = v3(0.0, 0.0, 0.0);

    pub fn dot(self, o: Vec3) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Vec3) -> Vec3 {
        v3(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    pub fn length_sq(self) -> f64 {
        self.dot(self)
    }

    pub fn distance(self, o: Vec3) -> f64 {
        (self - o).length()
    }

    pub fn distance_sq(self, o: Vec3) -> f64 {
        (self - o).length_sq()
    }

    /// Returns the unit vector, or `fallback` when the input is degenerate.
    pub fn normalize_or(self, fallback: Vec3) -> Vec3 {
        let l = self.length();
        if l > 1e-300 { self / l } else { fallback }
    }

    pub fn normalize(self) -> Vec3 {
        self.normalize_or(v3(0.0, 0.0, 1.0))
    }

    pub fn min(self, o: Vec3) -> Vec3 {
        v3(self.x.min(o.x), self.y.min(o.y), self.z.min(o.z))
    }

    pub fn max(self, o: Vec3) -> Vec3 {
        v3(self.x.max(o.x), self.y.max(o.y), self.z.max(o.z))
    }

    pub fn max_element(self) -> f64 {
        self.x.max(self.y).max(self.z)
    }

    pub fn is_finite(self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    pub fn as_f32(self) -> [f32; 3] {
        [self.x as f32, self.y as f32, self.z as f32]
    }

    /// Any unit vector perpendicular to `self`, chosen to avoid cancellation.
    pub fn any_perpendicular(self) -> Vec3 {
        let a = if self.x.abs() < 0.9 {
            v3(1.0, 0.0, 0.0)
        } else {
            v3(0.0, 1.0, 0.0)
        };
        self.cross(a).normalize()
    }
}

impl Add for Vec3 {
    type Output = Vec3;
    fn add(self, o: Vec3) -> Vec3 {
        v3(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl AddAssign for Vec3 {
    fn add_assign(&mut self, o: Vec3) {
        *self = *self + o;
    }
}

impl Sub for Vec3 {
    type Output = Vec3;
    fn sub(self, o: Vec3) -> Vec3 {
        v3(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Mul<f64> for Vec3 {
    type Output = Vec3;
    fn mul(self, s: f64) -> Vec3 {
        v3(self.x * s, self.y * s, self.z * s)
    }
}

impl Div<f64> for Vec3 {
    type Output = Vec3;
    fn div(self, s: f64) -> Vec3 {
        v3(self.x / s, self.y / s, self.z / s)
    }
}

impl Neg for Vec3 {
    type Output = Vec3;
    fn neg(self) -> Vec3 {
        v3(-self.x, -self.y, -self.z)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec2 {
    pub x: f64,
    pub y: f64,
}

pub const fn v2(x: f64, y: f64) -> Vec2 {
    Vec2 { x, y }
}

impl Add for Vec2 {
    type Output = Vec2;
    fn add(self, o: Vec2) -> Vec2 {
        v2(self.x + o.x, self.y + o.y)
    }
}

impl Sub for Vec2 {
    type Output = Vec2;
    fn sub(self, o: Vec2) -> Vec2 {
        v2(self.x - o.x, self.y - o.y)
    }
}

impl Mul<f64> for Vec2 {
    type Output = Vec2;
    fn mul(self, s: f64) -> Vec2 {
        v2(self.x * s, self.y * s)
    }
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy)]
pub struct Aabb {
    pub min: Vec3,
    pub max: Vec3,
}

impl Default for Aabb {
    fn default() -> Self {
        Aabb::EMPTY
    }
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb {
        min: v3(f64::INFINITY, f64::INFINITY, f64::INFINITY),
        max: v3(f64::NEG_INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY),
    };

    pub fn add_point(&mut self, p: Vec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }

    pub fn union(self, o: Aabb) -> Aabb {
        Aabb {
            min: self.min.min(o.min),
            max: self.max.max(o.max),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x
    }

    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// Longest edge, or 1.0 for an empty/degenerate box so callers can safely
    /// use it as a length scale.
    pub fn extent(&self) -> f64 {
        if self.is_empty() {
            return 1.0;
        }
        let d = self.size().max_element();
        if d > 1e-12 { d } else { 1.0 }
    }
}

/// Column-major 4x4 matrix, laid out for direct upload to WGSL.
#[derive(Debug, Clone, Copy)]
pub struct Mat4(pub [[f32; 4]; 4]);

impl Mat4 {
    /// Right-handed look-at view matrix.
    pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
        let f = (target - eye).normalize_or(v3(0.0, 0.0, -1.0));
        let mut s = f.cross(up);
        if s.length_sq() < 1e-20 {
            // Eye direction parallel to `up`: pick any stable right vector.
            s = f.any_perpendicular();
        }
        let s = s.normalize();
        let u = s.cross(f);
        Mat4([
            [s.x as f32, u.x as f32, -f.x as f32, 0.0],
            [s.y as f32, u.y as f32, -f.y as f32, 0.0],
            [s.z as f32, u.z as f32, -f.z as f32, 0.0],
            [
                -s.dot(eye) as f32,
                -u.dot(eye) as f32,
                f.dot(eye) as f32,
                1.0,
            ],
        ])
    }

    /// Perspective projection mapping to wgpu's 0..1 depth range.
    pub fn perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let t = 1.0 / (fov_y_rad * 0.5).tan();
        Mat4([
            [t / aspect, 0.0, 0.0, 0.0],
            [0.0, t, 0.0, 0.0],
            [0.0, 0.0, far / (near - far), -1.0],
            [0.0, 0.0, near * far / (near - far), 0.0],
        ])
    }

    /// The same projection with depth reversed: the near plane maps to 1 and
    /// the far plane to 0.
    ///
    /// Perspective crowds depth towards the far end, where a float is
    /// coarsest; reversed, the two cancel and depth resolves to about the same
    /// fraction of distance everywhere. With a near plane very close, the
    /// standard mapping is coarser than the bias that keeps edge lines in
    /// front of their faces.
    pub fn perspective_reversed(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
        let t = 1.0 / (fov_y_rad * 0.5).tan();
        Mat4([
            [t / aspect, 0.0, 0.0, 0.0],
            [0.0, t, 0.0, 0.0],
            [0.0, 0.0, near / (far - near), -1.0],
            [0.0, 0.0, near * far / (far - near), 0.0],
        ])
    }

    /// Right-handed orthographic projection about the view axis, mapping to
    /// wgpu's 0..1 depth range. Used for the shadow map: a directional light has
    /// no station point, so its rays are parallel and its projection is a box.
    pub fn orthographic(half_w: f32, half_h: f32, near: f32, far: f32) -> Mat4 {
        Mat4([
            [1.0 / half_w, 0.0, 0.0, 0.0],
            [0.0, 1.0 / half_h, 0.0, 0.0],
            [0.0, 0.0, 1.0 / (near - far), 0.0],
            [0.0, 0.0, near / (near - far), 1.0],
        ])
    }

    pub fn mul(&self, o: &Mat4) -> Mat4 {
        let mut r = [[0.0f32; 4]; 4];
        for c in 0..4 {
            for row in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += self.0[k][row] * o.0[c][k];
                }
                r[c][row] = s;
            }
        }
        Mat4(r)
    }
}
