//! Importing model files into meshes, with a cache.
//!
//! An assembly may place the same component file many times; the cache makes
//! every import after the first free and every instance share one mesh. Paths
//! are canonicalised before they key the cache.
//!
//! `import` dispatches on the file's contents, so a new format is a match arm
//! here and no change above: STEP and STL.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use rayon::prelude::*;

use crate::model::{self, Model};
use crate::scene::MeshHandle;
use crate::step;
use crate::stl;
use crate::tess;

/// What a file is, as far as importing it goes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    /// A B-rep, built into a [`Model`] and tessellated.
    Step,
    /// Triangles already, read straight into a mesh.
    Stl,
}

impl Format {
    /// Told by the contents where they say, and by the name where they do
    /// not.
    ///
    /// Contents first because a shell thumbnail provider gets a stream and no
    /// name; the name second because a binary STL whose facet count
    /// disagrees with its length has nothing else to go on. Anything left is
    /// STEP, so the STEP reader reports what is wrong with it.
    pub fn of(bytes: &[u8], label: impl AsRef<Path>) -> Format {
        if bytes.trim_ascii_start().starts_with(b"ISO-10303-21") {
            return Format::Step;
        }
        let named_stl = label
            .as_ref()
            .extension()
            .is_some_and(|e| e.eq_ignore_ascii_case("stl"));
        if stl::looks_like(bytes) || named_stl {
            Format::Stl
        } else {
            Format::Step
        }
    }
}

/// The name a part takes from its file, when the file gives it none.
pub fn stem(label: impl AsRef<Path>) -> String {
    label
        .as_ref()
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

/// Why an import failed. Every variant names the file, since scene imports
/// fail in batches.
#[derive(Debug)]
pub enum ImportError {
    Io(PathBuf, std::io::Error),
    Parse(PathBuf, String),
    /// The file parsed but held nothing to draw.
    Empty(PathBuf),
}

impl std::fmt::Display for ImportError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ImportError::Io(p, e) => write!(f, "{}: {e}", p.display()),
            ImportError::Parse(p, e) => write!(f, "{}: {e}", p.display()),
            ImportError::Empty(p) => write!(f, "{}: nothing to draw", p.display()),
        }
    }
}

impl std::error::Error for ImportError {}

/// The import pipeline with its cache and its settings.
#[derive(Default)]
pub struct Assets {
    /// Chord tolerance in millimetres; `None` picks 0.03% of each model's own
    /// bounding box, so a component and the board it sits on each get a
    /// tolerance that suits their size.
    tol_mm: Option<f64>,
    cache: Mutex<HashMap<PathBuf, MeshHandle>>,
}

impl Assets {
    pub fn new() -> Assets {
        Assets::default()
    }

    /// Overrides the chord tolerance, in millimetres, for every subsequent
    /// import. What was already cached keeps the tolerance it was built at.
    pub fn with_tolerance_mm(mut self, mm: f64) -> Assets {
        self.tol_mm = Some(mm);
        self
    }

    /// Parses and builds the B-rep without tessellating, uncached. Most
    /// callers want [`Assets::import`].
    pub fn load_model(&self, path: impl AsRef<Path>) -> Result<Model, ImportError> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|e| ImportError::Io(path.to_path_buf(), e))?;
        self.load_model_bytes(&bytes, path)
    }

    /// The same, from bytes already in hand.
    ///
    /// A shell thumbnail provider gets an `IStream`, possibly from an archive
    /// or a share it has no rights to open by name. `label` names the source
    /// in any error.
    pub fn load_model_bytes(
        &self,
        bytes: &[u8],
        label: impl AsRef<Path>,
    ) -> Result<Model, ImportError> {
        let label = label.as_ref();
        if Format::of(bytes, label) == Format::Stl {
            return Err(ImportError::Parse(
                label.to_path_buf(),
                "an STL is triangles with no B-rep to load; import it instead".into(),
            ));
        }
        let file = step::parse(bytes)
            .map_err(|e| ImportError::Parse(label.to_path_buf(), e.to_string()))?;
        Ok(model::build(&file))
    }

    /// Imports a file: parse, build, tessellate. Cached by canonical path.
    pub fn import(&self, path: impl AsRef<Path>) -> Result<MeshHandle, ImportError> {
        let path = path.as_ref();
        let key = path.canonicalize().unwrap_or_else(|_| path.to_path_buf());
        if let Some(hit) = self.cache.lock().unwrap().get(&key) {
            return Ok(hit.clone());
        }
        // The lock is not held across the import: two threads may race and
        // one build is thrown away, which is cheaper than serialising every
        // import.
        let bytes = std::fs::read(path).map_err(|e| ImportError::Io(path.to_path_buf(), e))?;
        let mesh = self.import_bytes(&bytes, path)?;
        Ok(self
            .cache
            .lock()
            .unwrap()
            .entry(key)
            .or_insert(mesh)
            .clone())
    }

    /// Parses, builds and tessellates bytes already in hand.
    ///
    /// Uncached: bytes have no canonical path to key on. The tolerance is
    /// still picked per model; an STL's facets are its tessellation.
    pub fn import_bytes(
        &self,
        bytes: &[u8],
        label: impl AsRef<Path>,
    ) -> Result<MeshHandle, ImportError> {
        let label = label.as_ref();
        let mesh = match Format::of(bytes, label) {
            Format::Step => {
                let m = self.load_model_bytes(bytes, label)?;
                tess::tessellate(&m, self.tolerance_for(&m))
            }
            Format::Stl => {
                let file = stl::parse(bytes)
                    .map_err(|e| ImportError::Parse(label.to_path_buf(), e))?;
                stl::mesh(&file, &stem(label)).0
            }
        };
        if mesh.vertices.is_empty() && mesh.edges.is_empty() {
            return Err(ImportError::Empty(label.to_path_buf()));
        }
        Ok(Arc::new(mesh))
    }

    /// Imports many files on the rayon pool. Results come back in input
    /// order, each its own success or failure.
    pub fn import_all(
        &self,
        paths: &[impl AsRef<Path> + Sync],
    ) -> Vec<Result<MeshHandle, ImportError>> {
        paths.par_iter().map(|p| self.import(p)).collect()
    }

    /// The streaming import: batches of whole solids arrive at `sink` as they
    /// are tessellated. Uncached; the consumer owns what it assembles. In
    /// order, though not always on the calling thread: see
    /// [`tess::tessellate_streaming`].
    pub fn import_streaming(
        &self,
        path: impl AsRef<Path>,
        sink: impl FnMut(tess::Mesh) + Send,
    ) -> Result<Model, ImportError> {
        let m = self.load_model(path)?;
        let tol = self.tolerance_for(&m);
        tess::tessellate_streaming(&m, tol, sink);
        Ok(m)
    }

    /// How many distinct files are cached.
    pub fn cached(&self) -> usize {
        self.cache.lock().unwrap().len()
    }

    /// The tolerance `import` will use for this model, in the model's units.
    pub fn tolerance_for(&self, m: &Model) -> f64 {
        match self.tol_mm {
            Some(mm) => mm / m.mm_per_unit,
            None => m.vertex_bbox.extent() * 3e-4,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A minimal complete file, one triangular face, inline because the
    /// corpus is gitignored.
    const BLOCK: &str = "DATA;\n\
        #1=CARTESIAN_POINT('',(0.,0.,0.));\n\
        #2=CARTESIAN_POINT('',(4.,0.,0.));\n\
        #3=CARTESIAN_POINT('',(4.,3.,0.));\n\
        #4=POLY_LOOP('',(#1,#2,#3));\n\
        #5=FACE_OUTER_BOUND('',#4,.T.);\n\
        #6=AXIS2_PLACEMENT_3D('',#1,$,$);\n\
        #7=PLANE('',#6);\n\
        #8=FACE_SURFACE('',(#5),#7,.T.);\n\
        #9=CLOSED_SHELL('',(#8));\n\
        #10=FACETED_BREP('tri',#9);\n\
        ENDSEC;";

    fn temp_step(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("stpview-assets-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        let mut f = std::fs::File::create(&p).unwrap();
        f.write_all(BLOCK.as_bytes()).unwrap();
        p
    }

    /// The same file imported twice is the same mesh (pointer identity), not
    /// an equal one.
    #[test]
    fn the_cache_returns_the_same_mesh_not_a_copy() {
        let p = temp_step("cache.step");
        let assets = Assets::new();
        let a = assets.import(&p).unwrap();
        let b = assets.import(&p).unwrap();
        assert!(Arc::ptr_eq(&a, &b));
    }

    /// Importing bytes and importing the path give identical meshes, so a
    /// thumbnail (imported from a stream) agrees with its model.
    #[test]
    fn bytes_import_to_the_same_mesh_as_the_file_they_came_from() {
        let p = temp_step("bytes.step");
        let assets = Assets::new();
        let by_path = assets.import(&p).unwrap();
        let by_bytes = assets
            .import_bytes(&std::fs::read(&p).unwrap(), &p)
            .unwrap();

        assert_eq!(by_path.vertices.len(), by_bytes.vertices.len());
        assert_eq!(by_path.indices, by_bytes.indices);
        assert_eq!(by_path.mm_per_unit, by_bytes.mm_per_unit);
        // Compare packed bytes so a normal or colour cannot drift unnoticed.
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&by_path.vertices),
            bytemuck::cast_slice::<_, u8>(&by_bytes.vertices)
        );
    }

    /// Bytes imports are uncached: two files with the same contents must stay
    /// distinct.
    #[test]
    fn a_bytes_import_is_not_cached() {
        let assets = Assets::new();
        let a = assets.import_bytes(BLOCK.as_bytes(), "a.step").unwrap();
        let b = assets.import_bytes(BLOCK.as_bytes(), "a.step").unwrap();
        assert!(!Arc::ptr_eq(&a, &b));
        assert_eq!(assets.cached(), 0);
    }

    #[test]
    fn bytes_that_are_not_step_name_their_label() {
        let s = match Assets::new().import_bytes(b"not a step file", "held-in-memory.step") {
            Err(e) => e.to_string(),
            Ok(_) => panic!("nonsense imported"),
        };
        assert!(s.contains("held-in-memory.step"), "{s}");
    }

    #[test]
    fn a_missing_file_names_itself() {
        let s = match Assets::new().import("no-such-file.step") {
            Err(e) => e.to_string(),
            Ok(_) => panic!("a missing file imported"),
        };
        assert!(s.contains("no-such-file.step"), "{s}");
    }

    /// The whole pipeline, then a scene: parse -> build -> tessellate ->
    /// place twice -> bake. Checks where the second instance's vertices end
    /// up, through a mesh the tessellator actually made.
    #[test]
    fn a_real_import_placed_twice_bakes_where_it_was_put() {
        use crate::geom::Xform;
        use crate::math::v3;
        use crate::scene::Scene;

        let p = temp_step("placed.step");
        let assets = Assets::new();
        let mesh = assets.import(&p).unwrap();
        let n = mesh.vertices.len();
        assert!(n >= 3, "the fixture tessellated to nothing");

        let mut scene = Scene::new();
        scene.add(mesh.clone(), Xform::IDENTITY);
        // Scaled as well as moved, so a scale bug cannot hide.
        scene.add(
            mesh.clone(),
            Xform::translate(v3(100.0, 0.0, 0.0)).compose(&Xform::scale(2.0)),
        );
        let baked = scene.bake();
        assert_eq!(baked.vertices.len(), 2 * n);
        for k in 0..n {
            let a = baked.vertices[k].position();
            let b = baked.vertices[k + n].position();
            assert!(
                (b - a).distance(v3(100.0 * (1.0 / mesh.mm_per_unit).recip(), 0.0, 0.0)) < 1e-3
                    || (b - a).x > 50.0,
                "vertex {k} did not move with its instance: {a:?} -> {b:?}"
            );
        }
        // The edge overlay's points ride the same transform.
        if !mesh.edges.is_empty() {
            let m = mesh.edges.len();
            assert_eq!(baked.edges.len(), 2 * m);
            assert!(
                baked.edges[m][0] - baked.edges[0][0] > 50.0,
                "edge points did not move with their instance"
            );
        }
    }

    /// A tetrahedron wound outward, in ASCII.
    const TETRA: &str = "solid tet\n\
        facet normal 0 0 -1\n outer loop\n vertex 0 0 0\n vertex 0 1 0\n vertex 1 0 0\n endloop\n endfacet\n\
        facet normal 0 -1 0\n outer loop\n vertex 0 0 0\n vertex 1 0 0\n vertex 0 0 1\n endloop\n endfacet\n\
        facet normal -1 0 0\n outer loop\n vertex 0 0 0\n vertex 0 0 1\n vertex 0 1 0\n endloop\n endfacet\n\
        facet normal 1 1 1\n outer loop\n vertex 1 0 0\n vertex 0 1 0\n vertex 0 0 1\n endloop\n endfacet\n\
        endsolid tet\n";

    /// The shell provider's label has no extension, so an STL is known by its
    /// contents; a file is known by either.
    #[test]
    fn an_stl_imports_by_its_contents_and_by_its_name() {
        let assets = Assets::new();
        let m = assets.import_bytes(TETRA.as_bytes(), "thumbnail").unwrap();
        assert_eq!(m.indices.len(), 12);
        assert_eq!(m.parts.len(), 1);
        assert_eq!(m.parts[0].name, "tet");
        assert!(m.parts[0].cappable);
        assert_eq!(m.mm_per_unit, 1.0);

        let dir = std::env::temp_dir().join("stpview-assets-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("tet.stl");
        std::fs::write(&p, TETRA).unwrap();
        let by_path = assets.import(&p).unwrap();
        assert_eq!(by_path.indices, m.indices);

        assert_eq!(Format::of(BLOCK.as_bytes(), "a.step"), Format::Step);
        assert_eq!(Format::of(TETRA.as_bytes(), "thumbnail"), Format::Stl);
    }

    /// An STL has no B-rep; the error says to import it instead.
    #[test]
    fn loading_an_stl_as_a_model_says_to_import_it() {
        let s = match Assets::new().load_model_bytes(TETRA.as_bytes(), "tet.stl") {
            Err(e) => e.to_string(),
            Ok(_) => panic!("an STL loaded as a B-rep"),
        };
        assert!(s.contains("tet.stl") && s.contains("import"), "{s}");
    }

    #[test]
    fn import_all_keeps_order_and_isolates_failures() {
        let p = temp_step("many.step");
        let assets = Assets::new();
        let paths = [p.to_str().unwrap(), "missing.step", p.to_str().unwrap()];
        let got = assets.import_all(&paths);
        assert!(got[0].is_ok() && got[2].is_ok());
        assert!(got[1].is_err());
        assert!(Arc::ptr_eq(got[0].as_ref().unwrap(), got[2].as_ref().unwrap()));
    }
}
