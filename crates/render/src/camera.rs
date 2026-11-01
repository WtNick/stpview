//! Orbit camera. Z-up, the usual convention of mechanical CAD.

use crate::math::{v3, Aabb, Mat4, Vec3};
use crate::tess::ViewTolerance;

/// Six-axis motion, normalised to -1..1 and already shaped for control.
///
/// Device frame: +X right, +Y forward (away from you), +Z up, rotations
/// right-handed about those axes. A SpaceMouse is one producer; anything that
/// can express "slide, push, twist" can drive [`Camera::fly`].
#[derive(Default, Clone, Copy, PartialEq, Debug)]
pub struct Motion {
    pub translate: [f64; 3],
    pub rotate: [f64; 3],
}

impl Motion {
    /// True when every axis is centred, so nothing would move.
    pub fn is_still(&self) -> bool {
        self.translate.iter().chain(&self.rotate).all(|&x| x == 0.0)
    }
}

/// Rates at full puck deflection, per second. Rotation in radians, pan in
/// viewport heights, zoom in the same steps the mouse wheel delivers.
///
/// A quarter turn a second keeps a shape legible while it moves; the pan is
/// matched to it.
const FLY_ORBIT: f64 = 1.6;
const FLY_PAN: f64 = 1.1;
const FLY_ZOOM: f64 = 12.0;
/// Slower than the orbit: roll is easy to catch by accident and has no
/// horizon to restore it.
const FLY_ROLL: f64 = 1.0;

pub struct Camera {
    pub target: Vec3,
    pub distance: f64,
    pub yaw: f64,
    pub pitch: f64,
    /// Rotation of the camera about its own view direction.
    ///
    /// Zero puts the world's up on the screen's up; only the SpaceMouse
    /// changes it. It is kept exactly zero, not merely small, because
    /// `basis` and `view` branch on it and `models/visual.ps1` allows no
    /// moved pixel.
    pub roll: f64,
    pub fov_y: f32,
    /// Where the model is and how big, used to keep the depth range sane.
    /// The depth range is measured from the eye to the model, not to the
    /// orbit target, which panning moves off the model.
    pub scene_radius: f64,
    pub scene_center: Vec3,
}

impl Camera {
    pub fn new(bbox: &Aabb) -> Camera {
        let mut c = Camera {
            target: Vec3::ZERO,
            distance: 1.0,
            yaw: -0.9,
            pitch: 0.5,
            roll: 0.0,
            fov_y: 45f32.to_radians(),
            scene_radius: 1.0,
            scene_center: Vec3::ZERO,
        };
        c.fit(bbox);
        c
    }

    /// Frames the whole bounding box without changing the viewing direction,
    /// except that it levels the horizon: an accidental roll has no other
    /// reset.
    pub fn fit(&mut self, bbox: &Aabb) {
        if bbox.is_empty() {
            return;
        }
        self.roll = 0.0;
        self.target = bbox.center();
        self.scene_center = bbox.center();
        // Enclosing sphere of the box, then back off far enough for the
        // narrower of the two field-of-view angles.
        self.scene_radius = bbox.size().length() * 0.5;
        if self.scene_radius < 1e-9 {
            self.scene_radius = 1.0;
        }
        self.distance = self.scene_radius / (self.fov_y as f64 * 0.5).sin() * 1.1;
    }

    /// Pulls in until the box itself fills a square frame, after
    /// [`Camera::fit`] has pointed at it.
    ///
    /// `fit` encloses the bounding sphere so the model stays in frame through
    /// an orbit; a tile or thumbnail is drawn from one angle, and a sphere
    /// round a long thin part is mostly empty. So the eight corners are
    /// projected and the distance taken from the most demanding one.
    ///
    /// Square, because `tan(fov_x/2) = aspect * tan(fov_y/2)`: one half-angle
    /// serves both axes only at aspect 1. Only `distance` moves; the scene
    /// centre and radius stay as the depth range's reference.
    pub fn close_in(&mut self, bbox: &Aabb) {
        if bbox.is_empty() {
            return;
        }
        let (right, up, forward) = self.basis();
        let t = (self.fov_y as f64 * 0.5).tan();
        let mut need: f64 = 0.0;
        for k in 0..8 {
            let c = v3(
                if k & 1 == 0 { bbox.min.x } else { bbox.max.x },
                if k & 2 == 0 { bbox.min.y } else { bbox.max.y },
                if k & 4 == 0 { bbox.min.z } else { bbox.max.z },
            ) - self.target;
            // Depth from the eye is `distance + z`, so a corner is inside the
            // frustum when |x| <= t * (distance + z).
            let z = c.dot(forward);
            need = need
                .max(c.dot(right).abs() / t - z)
                .max(c.dot(up).abs() / t - z);
        }
        // A little air, so nothing touches the edge of the frame.
        let want = need * 1.06;
        if want.is_finite() && want > 1e-9 {
            self.distance = want;
        }
    }

    pub fn eye(&self) -> Vec3 {
        let (cp, sp) = (self.pitch.cos(), self.pitch.sin());
        let (cy, sy) = (self.yaw.cos(), self.yaw.sin());
        self.target + v3(cp * cy, cp * sy, sp) * self.distance
    }

    /// Orthonormal camera basis: (right, up, forward).
    pub fn basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = (self.target - self.eye()).normalize_or(v3(0.0, 1.0, 0.0));
        let world_up = v3(0.0, 0.0, 1.0);
        let mut right = forward.cross(world_up);
        if right.length_sq() < 1e-12 {
            right = forward.any_perpendicular();
        }
        let right = right.normalize();
        // Rodrigues about the view direction, branched on zero: renormalising
        // a length that is 1.0 only to within rounding would move pixels in
        // the stored views.
        let right = if self.roll == 0.0 {
            right
        } else {
            (right * self.roll.cos() + forward.cross(right) * self.roll.sin()).normalize()
        };
        (right, right.cross(forward), forward)
    }

    pub fn orbit(&mut self, dx: f64, dy: f64) {
        const LIMIT: f64 = std::f64::consts::FRAC_PI_2 - 1e-3;
        self.yaw -= dx;
        self.pitch = (self.pitch + dy).clamp(-LIMIT, LIMIT);
    }

    /// Pans by a screen-space offset given in fractions of the viewport height.
    pub fn pan(&mut self, dx: f64, dy: f64) {
        let (right, up, _) = self.basis();
        // Scale so a drag moves the point under the cursor with the cursor.
        let world_per_unit = 2.0 * self.distance * (self.fov_y as f64 * 0.5).tan();
        self.target = self.target + right * (-dx * world_per_unit) + up * (dy * world_per_unit);
    }

    pub fn zoom(&mut self, steps: f64) {
        self.distance *= (-steps * 0.12).exp();
        let min = self.scene_radius * 1e-3;
        let max = self.scene_radius * 200.0;
        self.distance = self.distance.clamp(min, max);
    }

    /// Applies one time step of six-axis motion from a SpaceMouse.
    ///
    /// | Puck | Camera |
    /// | --- | --- |
    /// | slide left/right | pan across the screen |
    /// | push/pull forward | zoom in and out |
    /// | lift/press down | pan up and down |
    /// | tip the far edge down | orbit up over the model |
    /// | twist | orbit around it |
    /// | roll | roll the camera about the view direction |
    ///
    /// Scaled by `dt`, so speed depends on how long the puck is held, not on
    /// report or frame rate.
    ///
    /// **Every sign is in this one function** and was measured on the
    /// hardware, not derived; `--puck` prints the axes.
    pub fn fly(&mut self, m: &Motion, dt: f64) {
        if m.is_still() || dt <= 0.0 {
            return;
        }
        let [tx, ty, tz] = m.translate;
        let [pitch, roll, yaw] = m.rotate;

        // Pan takes fractions of the viewport height and carries the model
        // with it: puck right slides the model right.
        if tx != 0.0 || tz != 0.0 {
            self.pan(tx * FLY_PAN * dt, tz * FLY_PAN * dt);
        }
        // Pushing the puck away from you goes further into the model.
        if ty != 0.0 {
            self.zoom(ty * FLY_ZOOM * dt);
        }
        // `orbit` subtracts its first argument from the yaw. Tipping the
        // puck's far edge down is negative about +X and raises the eye.
        if pitch != 0.0 || yaw != 0.0 {
            self.orbit(-yaw * FLY_ORBIT * dt, -pitch * FLY_ORBIT * dt);
        }
        // The camera turns against the puck, so the model turns with it.
        if roll != 0.0 {
            self.roll -= roll * FLY_ROLL * dt;
            // Wrapped so the f64 does not grow and lose precision.
            self.roll = self.roll.rem_euclid(std::f64::consts::TAU);
        }
    }

    /// Near and far planes chosen to bracket the model.
    ///
    /// The near plane is a fifth of the gap between the eye and the model's
    /// bounding sphere, measured from the eye to the model, never to the
    /// orbit target: after `pan` or `--view <at>`, `distance` says nothing
    /// about where the geometry is. A near plane that clips geometry also
    /// breaks the section cap, whose parity is counted with no depth test.
    pub fn depth_range(&self) -> (f32, f32) {
        // Both ends from the eye's distance, so two cameras with the same eye
        // and lens get the same depth buffer whatever their orbit distance.
        let reach = self.eye().distance(self.scene_center);
        let far = (reach + self.scene_radius * 2.0).max(self.scene_radius * 0.01);
        let gap = (reach - self.scene_radius).max(0.0);
        // Keep enough depth precision when zoomed close to a big model.
        let near = (gap * 0.2)
            .max(far * 1e-5)
            .max(self.scene_radius * 1e-5);
        (near as f32, far as f32)
    }

    /// The tessellation this view needs: within its frustum, a chord may stray
    /// `px` pixels of a `width` by `height` picture at its own depth, and never
    /// less than `floor`. Outside the frustum the view asks nothing.
    ///
    /// The frustum is widened by a sixth each way, so nudging the camera does
    /// not bring coarse tessellation in at the edge of the picture.
    pub fn view_tolerance(&self, width: u32, height: u32, px: f64, floor: f64) -> ViewTolerance {
        let (right, up, forward) = self.basis();
        let t = (self.fov_y as f64 * 0.5).tan();
        let (w, h) = (width.max(1) as f64, height.max(1) as f64);
        const MARGIN: f64 = 7.0 / 6.0;
        ViewTolerance {
            eye: self.eye(),
            right,
            up,
            forward,
            tan_x: t * w / h * MARGIN,
            tan_y: t * MARGIN,
            // A pixel at unit depth: the view's height there (twice the
            // tangent) over the pixel count.
            per_depth: px * 2.0 * t / h,
            near: self.depth_range().0 as f64,
            floor,
        }
    }

    pub fn view(&self) -> Mat4 {
        // Unrolled, the world's up goes in unchanged: `basis`'s up is equal
        // mathematically but differs in the last bits, which would move every
        // stored view.
        let up = if self.roll == 0.0 {
            v3(0.0, 0.0, 1.0)
        } else {
            self.basis().1
        };
        Mat4::look_at(self.eye(), self.target, up)
    }

    /// With depth reversed, near at 1 and far at 0: see
    /// `Mat4::perspective_reversed`, and `render::NEARER` for what the depth
    /// tests do with it.
    pub fn proj(&self, aspect: f32) -> Mat4 {
        let (near, far) = self.depth_range();
        Mat4::perspective_reversed(self.fov_y, aspect.max(1e-3), near, far)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj(aspect).mul(&self.view())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn box_of(min: Vec3, max: Vec3) -> Aabb {
        let mut b = Aabb::EMPTY;
        b.add_point(min);
        b.add_point(max);
        b
    }

    /// How much of a square frame the box fills, or `None` if any corner falls
    /// outside it. 1.0 means a corner sits exactly on the edge.
    fn frame_fill(c: &Camera, b: &Aabb) -> Option<f32> {
        let vp = c.view_proj(1.0);
        let mut fill: f32 = 0.0;
        for i in 0..8 {
            let p = v3(
                if i & 1 == 0 { b.min.x } else { b.max.x },
                if i & 2 == 0 { b.min.y } else { b.max.y },
                if i & 4 == 0 { b.min.z } else { b.max.z },
            );
            let m = &vp.0;
            let clip: Vec<f32> = (0..4)
                .map(|r| m[0][r] * p.x as f32 + m[1][r] * p.y as f32 + m[2][r] * p.z as f32 + m[3][r])
                .collect();
            let w = clip[3];
            if !(w > 0.0) {
                return None;
            }
            let (x, y) = (clip[0].abs() / w, clip[1].abs() / w);
            if x > 1.0 || y > 1.0 {
                return None;
            }
            fill = fill.max(x).max(y);
        }
        Some(fill)
    }

    /// `close_in` fills the frame with the box without cropping it. A long
    /// thin box is the case `fit` wastes, its bounding sphere being mostly
    /// empty.
    #[test]
    fn close_in_fills_a_square_frame_without_cropping() {
        let b = box_of(v3(-50.0, -2.0, -2.0), v3(50.0, 2.0, 2.0));
        let fitted = Camera::new(&b);
        let mut closed = Camera::new(&b);
        closed.close_in(&b);

        assert!(
            closed.distance < fitted.distance,
            "close_in backed off instead of pulling in: {} vs {}",
            closed.distance,
            fitted.distance
        );
        let fitted_fill = frame_fill(&fitted, &b).expect("fit cropped the box");
        let fill = frame_fill(&closed, &b).expect("close_in cropped the box");
        // 1.06 of air was asked for, so the tightest corner sits at about
        // 0.94. The gain over `fit` is modest seen corner-on like this and
        // large when the long axis lies across the view.
        assert!(fill > 0.9, "the box fills only {fill} of the frame");
        assert!(
            fill > fitted_fill,
            "close_in gained nothing over fit: {fill} vs {fitted_fill}"
        );
    }

    /// `close_in` moves only `distance`; the depth range's reference (scene
    /// centre and radius) stays put.
    #[test]
    fn close_in_moves_nothing_but_the_distance() {
        let b = box_of(v3(-50.0, -2.0, -2.0), v3(50.0, 2.0, 2.0));
        let before = Camera::new(&b);
        let mut after = Camera::new(&b);
        after.close_in(&b);

        assert_eq!(after.target, before.target);
        assert_eq!(after.scene_center, before.scene_center);
        assert_eq!(after.scene_radius, before.scene_radius);
        assert_eq!(after.yaw, before.yaw);
        assert_eq!(after.pitch, before.pitch);
        assert_eq!(after.roll, before.roll);
    }

    /// An empty box leaves the camera alone rather than dividing by nothing;
    /// a model with no geometry still reaches a thumbnail.
    #[test]
    fn close_in_on_nothing_is_a_no_op() {
        let b = box_of(v3(-1.0, -1.0, -1.0), v3(1.0, 1.0, 1.0));
        let mut c = Camera::new(&b);
        let was = c.distance;
        c.close_in(&Aabb::EMPTY);
        assert_eq!(c.distance, was);
    }

    #[test]
    fn fit_frames_the_whole_box() {
        let b = box_of(v3(-10.0, -10.0, -10.0), v3(10.0, 10.0, 10.0));
        let c = Camera::new(&b);
        assert!(c.target.distance(Vec3::ZERO) < 1e-12);
        // Every corner must fall inside the view frustum.
        let vp = c.view_proj(1.0);
        for i in 0..8 {
            let p = v3(
                if i & 1 == 0 { -10.0 } else { 10.0 },
                if i & 2 == 0 { -10.0 } else { 10.0 },
                if i & 4 == 0 { -10.0 } else { 10.0 },
            );
            let m = &vp.0;
            let clip: Vec<f32> = (0..4)
                .map(|r| {
                    m[0][r] * p.x as f32
                        + m[1][r] * p.y as f32
                        + m[2][r] * p.z as f32
                        + m[3][r]
                })
                .collect();
            let w = clip[3];
            assert!(w > 0.0, "corner {i} behind the camera");
            assert!(clip[0].abs() <= w, "corner {i} outside horizontally");
            assert!(clip[1].abs() <= w, "corner {i} outside vertically");
            assert!((0.0..=w).contains(&clip[2]), "corner {i} outside depth");
        }
    }

    #[test]
    fn eye_stays_at_the_orbit_distance() {
        let c = Camera::new(&box_of(v3(0.0, 0.0, 0.0), v3(2.0, 4.0, 6.0)));
        assert!((c.eye().distance(c.target) - c.distance).abs() < 1e-9);
        let mut c2 = Camera::new(&box_of(v3(0.0, 0.0, 0.0), v3(2.0, 4.0, 6.0)));
        c2.orbit(1.3, -0.7);
        assert!((c2.eye().distance(c2.target) - c2.distance).abs() < 1e-9);
    }

    #[test]
    fn pitch_never_reaches_the_pole() {
        let mut c = Camera::new(&box_of(v3(0.0, 0.0, 0.0), v3(1.0, 1.0, 1.0)));
        for _ in 0..100 {
            c.orbit(0.0, 1.0);
        }
        assert!(c.pitch < std::f64::consts::FRAC_PI_2);
        // The basis must stay well conditioned at the limit.
        let (right, up, fwd) = c.basis();
        assert!((right.length() - 1.0).abs() < 1e-9);
        assert!(right.dot(fwd).abs() < 1e-9 && right.dot(up).abs() < 1e-9);
    }

    #[test]
    fn zoom_is_bounded() {
        let mut c = Camera::new(&box_of(v3(0.0, 0.0, 0.0), v3(10.0, 10.0, 10.0)));
        for _ in 0..500 {
            c.zoom(10.0);
        }
        assert!(c.distance > 0.0 && c.distance.is_finite());
        for _ in 0..1000 {
            c.zoom(-10.0);
        }
        assert!(c.distance.is_finite() && c.distance <= c.scene_radius * 200.0 + 1e-9);
    }

    /// Two cameras with the same eye and lens see the same depths, whatever
    /// their orbit distance. A near plane taken from the target distance
    /// clips geometry after a pan, and shows up as a wrong section cap.
    #[test]
    fn the_depth_range_does_not_move_with_the_orbit_target() {
        let b = box_of(v3(-100.0, -100.0, -100.0), v3(100.0, 100.0, 100.0));
        let base = Camera::new(&b);
        let eye = base.eye();
        let unit = (eye - base.target) / base.distance;
        let want = base.depth_range();
        for scale in [1.0, 2.0, 8.0, 40.0] {
            let mut c = Camera::new(&b);
            c.distance = base.distance * scale;
            // The same eye, reached from a target that much further back.
            c.target = eye - unit * c.distance;
            assert!(c.eye().distance(eye) < 1e-9, "the eye moved");
            let got = c.depth_range();
            assert!(
                (got.0 - want.0).abs() <= want.0 * 1e-4 && (got.1 - want.1).abs() <= want.1 * 1e-4,
                "orbiting {scale}x further gave {got:?}, not {want:?}"
            );
        }
    }

    /// The near plane never reaches the model, from anywhere: clipped
    /// geometry is indistinguishable downstream from missing geometry.
    #[test]
    fn the_near_plane_never_reaches_the_model() {
        let b = box_of(v3(-100.0, -50.0, -30.0), v3(100.0, 50.0, 30.0));
        for (yaw, pitch) in [(0.0, 0.0), (1.2, 0.6), (-2.0, -1.1)] {
            for target in [
                v3(0.0, 0.0, 0.0),
                v3(500.0, 0.0, 0.0),
                v3(-3000.0, 400.0, 200.0),
            ] {
                for distance in [1.0, 50.0, 300.0, 3000.0] {
                    let mut c = Camera::new(&b);
                    c.yaw = yaw;
                    c.pitch = pitch;
                    c.target = target;
                    c.distance = distance;
                    let (near, far) = c.depth_range();
                    assert!(near > 0.0 && far > near, "near {near} far {far}");
                    let gap = c.eye().distance(c.scene_center) - c.scene_radius;
                    if gap > 0.0 {
                        assert!(
                            (near as f64) < gap,
                            "near {near} clips into the model {gap} away"
                        );
                    }
                }
            }
        }
    }

    fn puck(translate: [f64; 3], rotate: [f64; 3]) -> Motion {
        Motion { translate, rotate }
    }

    fn test_camera() -> Camera {
        Camera::new(&box_of(v3(-10.0, -10.0, -10.0), v3(10.0, 10.0, 10.0)))
    }

    /// The SpaceMouse moves the camera by how long it is held, not by how many
    /// reports arrive or frames are drawn.
    ///
    /// Orbit and zoom compose exactly at any step, so they are held to the
    /// last bit. Pan follows a curve (the camera turns and dollies while it
    /// pans), so two real frame rates must agree and halving the step must
    /// converge.
    #[test]
    fn flying_depends_on_how_long_the_puck_is_held_not_on_the_frame_rate() {
        let m = puck([0.4, -0.3, 0.2], [0.25, 0.0, -0.35]);
        // One second of holding it, at a given frame rate.
        let second_at = |hz: u32| -> Camera {
            let mut c = test_camera();
            for _ in 0..hz {
                c.fly(&m, 1.0 / hz as f64);
            }
            c
        };

        let slow = second_at(60);
        let fast = second_at(144);
        assert!((slow.yaw - fast.yaw).abs() < 1e-12, "{} vs {}", slow.yaw, fast.yaw);
        assert!((slow.pitch - fast.pitch).abs() < 1e-12);
        assert!((slow.distance - fast.distance).abs() < 1e-12 * slow.distance);
        assert!(
            slow.target.distance(fast.target) < slow.scene_radius * 0.02,
            "60 Hz and 144 Hz disagree by more than 2% of the model: {:?} vs {:?}",
            slow.target,
            fast.target
        );

        // Converging, not merely close: each halving of the step must bring
        // the answer at least a third closer to the next.
        let mut gap = f64::INFINITY;
        for hz in [60u32, 120, 240, 480] {
            let next = second_at(hz * 2).target;
            let this = second_at(hz).target.distance(next);
            assert!(
                this < gap * 0.67,
                "refining from {hz} Hz gained too little: {this} against {gap}"
            );
            gap = this;
        }
    }

    /// Every sign, pinned to what was observed on the hardware rather than to
    /// a derivation, so a later tidy-up cannot quietly undo it.
    #[test]
    fn each_axis_moves_the_model_the_way_the_puck_was_observed_to() {
        let base = test_camera();
        let (right, up, _) = base.basis();

        // Slide the puck right: the model slides right, so the camera looks
        // further left.
        let mut c = test_camera();
        c.fly(&puck([1.0, 0.0, 0.0], [0.0; 3]), 0.1);
        assert!((c.target - base.target).dot(right) < 0.0, "right slide");

        // Lift the puck and the camera rises with it, so the model drops down
        // the screen. Unlike the horizontal slide, but it is what the device
        // does.
        let mut c = test_camera();
        c.fly(&puck([0.0, 0.0, 1.0], [0.0; 3]), 0.1);
        assert!((c.target - base.target).dot(up) > 0.0, "lift");

        // Push it away: into the model.
        let mut c = test_camera();
        c.fly(&puck([0.0, 1.0, 0.0], [0.0; 3]), 0.1);
        assert!(c.distance < base.distance, "push away should zoom in");

        // Tip the far edge down and rise over the top of the model.
        let mut c = test_camera();
        c.fly(&puck([0.0; 3], [-1.0, 0.0, 0.0]), 0.1);
        assert!(c.pitch > base.pitch, "tipping away should raise the eye");

        // Twist, as measured on the hardware.
        let mut c = test_camera();
        c.fly(&puck([0.0; 3], [0.0, 0.0, 1.0]), 0.1);
        assert!(c.yaw > base.yaw, "twist");

        // Roll turns the camera against the puck, so the model turns with it.
        let mut c = test_camera();
        c.fly(&puck([0.0; 3], [0.0, 1.0, 0.0]), 0.1);
        assert!(c.roll != 0.0, "roll did nothing");
        let (rolled_right, _, _) = c.basis();
        assert!(
            rolled_right.dot(up) > 0.0,
            "a positive roll should tip the screen's right edge up"
        );
    }

    /// Roll changes the view matrix, and `fit` levels it back to a frame
    /// bit-identical to an unrolled camera (`models/visual.ps1` allows no
    /// moved pixel).
    #[test]
    fn roll_reaches_the_view_and_fit_levels_it() {
        let b = box_of(v3(-10.0, -10.0, -10.0), v3(10.0, 10.0, 10.0));
        let level = test_camera();
        let mut tilted = test_camera();
        tilted.roll = 0.4;
        assert_ne!(level.view().0, tilted.view().0, "roll never reached the view");

        // The rolled frame stays orthonormal, or pan and the light rig would
        // drift as it turns.
        let (r, u, f) = tilted.basis();
        for (a, b) in [(r, u), (u, f), (f, r)] {
            assert!(a.dot(b).abs() < 1e-9, "basis not orthogonal");
        }
        for v in [r, u, f] {
            assert!((v.length() - 1.0).abs() < 1e-9, "basis not unit");
        }

        tilted.fit(&b);
        assert_eq!(tilted.roll, 0.0, "fit must level the horizon");
        assert_eq!(
            level.view().0,
            tilted.view().0,
            "a levelled camera must match one that was never rolled, bit for bit"
        );
    }

    /// Roll and a centred puck leave yaw, pitch, distance and target exactly
    /// alone; a leak would show as the view creeping under a resting hand.
    #[test]
    fn roll_and_a_centred_puck_move_nothing() {
        let base = test_camera();
        for m in [
            puck([0.0; 3], [0.0, 1.0, 0.0]),
            puck([0.0; 3], [0.0, -1.0, 0.0]),
            Motion::default(),
        ] {
            let mut c = test_camera();
            c.fly(&m, 0.5);
            assert_eq!(c.yaw, base.yaw);
            assert_eq!(c.pitch, base.pitch);
            assert_eq!(c.distance, base.distance);
            assert_eq!(c.target.x, base.target.x);
            assert_eq!(c.target.y, base.target.y);
            assert_eq!(c.target.z, base.target.z);
        }
    }

    /// Held at full deflection, the camera stays drawable: `fly` goes through
    /// the clamps in `zoom` and `orbit`, not round them.
    #[test]
    fn holding_the_puck_at_full_deflection_stays_bounded() {
        let mut c = test_camera();
        for _ in 0..2000 {
            c.fly(&puck([1.0, 1.0, 1.0], [1.0, 1.0, 1.0]), 0.05);
            assert!(c.distance.is_finite() && c.distance > 0.0);
            assert!(c.pitch.abs() < std::f64::consts::FRAC_PI_2);
            let (near, far) = c.depth_range();
            assert!(near > 0.0 && far > near, "near {near} far {far}");
        }
        assert!(c.distance <= c.scene_radius * 200.0 + 1e-9);
    }

    /// Depth is reversed, near at 1 and far at 0, and resolves an edge pulled
    /// towards the eye from its face at every depth in view, in f32 as the
    /// depth test does it. With the camera inside a long part the near plane
    /// is tiny, and conventional depth would break edges into dashes.
    #[test]
    fn the_depth_buffer_keeps_a_nudged_edge_in_front_of_its_face_at_every_depth() {
        let b = box_of(v3(0.0, -2.0, -2.0), v3(600.0, 2.0, 2.0));
        let mut c = Camera::new(&b);
        c.target = v3(520.0, 0.0, 0.0);
        c.distance = 60.0;
        c.yaw = (-30f64).to_radians();
        c.pitch = 20f64.to_radians();
        let (near, far) = c.depth_range();
        assert!(near < far * 1e-4, "the near plane is not close: {near} against {far}");

        let m = c.proj(1.0).0;
        // Depth at a distance straight ahead, as the rasteriser divides it.
        let depth = |d: f64| {
            let z = -(d as f32);
            (m[2][2] * z + m[3][2]) / (m[2][3] * z + m[3][3])
        };
        assert!((depth(near as f64) - 1.0).abs() < 1e-6, "near is not at 1");
        assert!(depth(far as f64).abs() < 1e-6, "far is not at 0");

        let nudge = crate::render::EDGE_EYE_NUDGE as f64;
        for k in 0..=64 {
            let d = near as f64 * (far as f64 / near as f64).powf(k as f64 / 64.0);
            let (face, edge) = (depth(d), depth(d * (1.0 - nudge)));
            assert!(
                edge > face,
                "at {d:.3} away an edge pulled {:.4} nearer is not in front of its face: {edge} against {face}",
                d * nudge
            );
        }
    }

    #[test]
    fn depth_range_stays_positive_and_ordered() {
        let mut c = Camera::new(&box_of(v3(-500.0, -500.0, -500.0), v3(500.0, 500.0, 500.0)));
        for _ in 0..40 {
            c.zoom(5.0);
            let (near, far) = c.depth_range();
            assert!(near > 0.0 && far > near, "near {near} far {far}");
        }
    }
}
