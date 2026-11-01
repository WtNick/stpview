//! Composition: imported meshes placed into one world.
//!
//! A scene is a list of instances -- which mesh, where, and optionally
//! tinted. The world's unit is the millimetre: each instance's transform is
//! composed with its mesh's `mm_per_unit`, so files authored in different
//! units land at the right sizes. Assembly placements from the file itself
//! are already inside the mesh.

use std::sync::Arc;

use crate::geom::Xform;
use crate::math::{v3, Aabb};
use crate::tess::Mesh;

/// A shared tessellated model: placing a mesh many times costs transforms,
/// not copies.
pub type MeshHandle = Arc<Mesh>;

/// One placed mesh.
#[derive(Clone)]
pub struct Instance {
    pub mesh: MeshHandle,
    /// Where it sits, in scene millimetres. The mesh's own unit scale is not
    /// in here; it is applied when the scene is realised.
    pub place: Xform,
    /// Optional colour multiplier, RGBA in 0..1. `None` leaves the imported
    /// colours alone.
    pub tint: Option<[f32; 4]>,
}

impl Instance {
    /// The transform this instance's geometry undergoes: the placement, then
    /// the mesh's unit scale inside it. Shared by `bounds`, `bake` and the
    /// instanced renderer so the camera frames exactly what is drawn.
    pub fn world(&self) -> Xform {
        self.place.compose(&Xform::scale(self.mesh.mm_per_unit))
    }
}

/// Names an instance in its scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InstanceId(pub usize);

#[derive(Default, Clone)]
pub struct Scene {
    pub instances: Vec<Instance>,
}

impl Scene {
    pub fn new() -> Scene {
        Scene::default()
    }

    /// Places a mesh. The transform is in scene millimetres.
    pub fn add(&mut self, mesh: MeshHandle, place: Xform) -> InstanceId {
        self.add_tinted(mesh, place, None)
    }

    pub fn add_tinted(&mut self, mesh: MeshHandle, place: Xform, tint: Option<[f32; 4]>) -> InstanceId {
        self.instances.push(Instance { mesh, place, tint });
        InstanceId(self.instances.len() - 1)
    }

    /// Bounds of everything placed, in scene millimetres.
    pub fn bounds(&self) -> Aabb {
        let mut bb = Aabb::EMPTY;
        for i in &self.instances {
            let w = i.world();
            let b = i.mesh.bbox;
            if b.is_empty() {
                continue;
            }
            for k in 0..8 {
                let x = if k & 1 == 0 { b.min.x } else { b.max.x };
                let y = if k & 2 == 0 { b.min.y } else { b.max.y };
                let z = if k & 4 == 0 { b.min.z } else { b.max.z };
                bb.add_point(w.point(v3(x, y, z)));
            }
        }
        bb
    }

    /// Realises the scene as one mesh, in millimetres, copying each
    /// instance's vertices. The reference the instanced renderer is checked
    /// against.
    pub fn bake(&self) -> Mesh {
        let mut out = Mesh {
            // The world is millimetres by definition.
            mm_per_unit: 1.0,
            ..Default::default()
        };
        for i in &self.instances {
            out.absorb_placed(&i.mesh, &i.world(), i.tint);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::Vec3;
    use crate::tess::{Mesh, Vertex};

    /// A one-triangle mesh with a distinctive size, for placement tests.
    fn triangle(mm_per_unit: f64) -> MeshHandle {
        let mut m = Mesh {
            mm_per_unit,
            ..Default::default()
        };
        let n = v3(0.0, 0.0, 1.0);
        m.vertices.push(Vertex::new(v3(0.0, 0.0, 0.0), n, [1.0, 0.0, 0.0], 1.0));
        m.vertices.push(Vertex::new(v3(2.0, 0.0, 0.0), n, [1.0, 0.0, 0.0], 1.0));
        m.vertices.push(Vertex::new(v3(0.0, 1.0, 0.0), n, [1.0, 0.0, 0.0], 1.0));
        m.indices.extend([0, 1, 2]);
        for v in &m.vertices {
            m.bbox.add_point(v.position());
        }
        Arc::new(m)
    }

    /// A mesh in inches lands 25.4 times larger; the translation, already in
    /// millimetres, is not scaled.
    #[test]
    fn unit_scale_is_inside_the_placement_not_outside_it() {
        let mut scene = Scene::new();
        scene.add(triangle(25.4), Xform::translate(v3(10.0, 0.0, 0.0)));
        let b = scene.bounds();
        assert!((b.max.x - (10.0 + 2.0 * 25.4)).abs() < 1e-9, "{:?}", b);
        assert!((b.min.x - 10.0).abs() < 1e-9);

        let baked = scene.bake();
        assert_eq!(baked.mm_per_unit, 1.0);
        let bb = baked.bbox;
        assert!((bb.max.x - (10.0 + 2.0 * 25.4)).abs() < 1e-6, "{:?}", bb);
    }

    /// Bounds frame exactly what bake builds.
    #[test]
    fn bounds_and_bake_agree() {
        let mut scene = Scene::new();
        scene.add(triangle(1.0), Xform::IDENTITY);
        scene.add(
            triangle(2.0),
            Xform::translate(v3(-3.0, 4.0, 5.0)).compose(&Xform::rotate_axis(2, 1.1)),
        );
        let said = scene.bounds();
        let built = scene.bake().bbox;
        assert!(said.min.distance(built.min) < 1e-6, "{said:?} vs {built:?}");
        assert!(said.max.distance(built.max) < 1e-6);
    }

    /// Two instances share one mesh; the bake holds two copies with disjoint
    /// ranges, and the second copy's vertices (not just its bounding box,
    /// which is transformed separately) actually moved.
    #[test]
    fn instances_share_the_mesh_and_bake_disjoint_ranges() {
        let tri = triangle(1.0);
        let mut scene = Scene::new();
        scene.add(tri.clone(), Xform::IDENTITY);
        scene.add(tri.clone(), Xform::translate(v3(5.0, 0.0, 0.0)));
        assert_eq!(Arc::strong_count(&tri), 3, "the scene must not deep-copy");

        let baked = scene.bake();
        assert_eq!(baked.vertices.len(), 6);
        assert_eq!(baked.indices.len(), 6);
        assert_eq!(baked.indices[3..], [3, 4, 5]);
        for k in 0..3 {
            let a = baked.vertices[k].position();
            let b = baked.vertices[k + 3].position();
            assert!(
                (b - a).distance(v3(5.0, 0.0, 0.0)) < 1e-6,
                "vertex {k} did not move with its instance: {a:?} -> {b:?}"
            );
        }
    }

    /// A mirrored instance swaps its winding: reflection reverses orientation,
    /// and the section cap's parity count depends on it.
    #[test]
    fn a_mirrored_instance_swaps_its_winding() {
        let tri = triangle(1.0);
        let mut scene = Scene::new();
        let mut mirror = Xform::IDENTITY;
        mirror.x = v3(-1.0, 0.0, 0.0);
        scene.add(tri, mirror);
        let baked = scene.bake();
        assert_eq!(baked.indices, vec![0, 2, 1]);
        // An x-mirror leaves +Z alone, so the normal still points up.
        let n = baked.vertices[0].normal();
        assert!(n.distance(Vec3 { x: 0.0, y: 0.0, z: 1.0 }) < 1e-3, "{n:?}");
    }

    /// Tint multiplies; absence leaves the imported colour byte-identical.
    #[test]
    fn tint_multiplies_and_none_is_exact() {
        let tri = triangle(1.0);
        let mut scene = Scene::new();
        scene.add(tri.clone(), Xform::IDENTITY);
        scene.add_tinted(tri.clone(), Xform::IDENTITY, Some([0.5, 1.0, 1.0, 1.0]));
        let baked = scene.bake();
        assert_eq!(baked.vertices[0].color_bytes(), tri.vertices[0].color_bytes());
        let tinted = baked.vertices[3].color_bytes();
        let plain = tri.vertices[0].color_bytes();
        assert_eq!(tinted[0], (plain[0] as f32 * 0.5).round() as u8);
        assert_eq!(tinted[1], plain[1]);
    }
}
