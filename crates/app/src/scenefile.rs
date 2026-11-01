//! The scene description file: a list of models and where to put them.
//!
//! ```toml
//! # All lengths in millimetres, rotations in degrees.
//! [[place]]
//! file = "board.step"
//!
//! [[place]]
//! file = "PinHeader_1x04.step"
//! at = [10.0, 5.0, 1.6]
//! rotate = [0.0, 0.0, 90.0]   # about X, then Y, then Z
//! tint = [0.9, 0.9, 0.9, 1.0]
//! ```
//!
//! A hand-written reader for this TOML subset -- quoted strings, arrays of
//! numbers, `[[place]]` -- rejecting anything else by line number, so a file
//! that works here works in any real TOML parser too.

use std::path::{Path, PathBuf};

use stpview::geom::Xform;
use stpview::math::v3;
use stpview::{Assets, Scene};

/// One `[[place]]` block, as written.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaceSpec {
    pub file: PathBuf,
    pub at: [f64; 3],
    /// Degrees about X, Y, Z, applied in that order.
    pub rotate: [f64; 3],
    /// Uniform, applied before the rotation: a scaled part turns as a rigid
    /// object, not into a sheared one.
    pub scale: f64,
    pub tint: Option<[f32; 4]>,
}

impl Default for PlaceSpec {
    fn default() -> PlaceSpec {
        PlaceSpec {
            file: PathBuf::new(),
            at: [0.0; 3],
            rotate: [0.0; 3],
            scale: 1.0,
            tint: None,
        }
    }
}

impl PlaceSpec {
    /// The transform this spec means, in scene millimetres.
    pub fn xform(&self) -> Xform {
        let rx = Xform::rotate_axis(0, self.rotate[0].to_radians());
        let ry = Xform::rotate_axis(1, self.rotate[1].to_radians());
        let rz = Xform::rotate_axis(2, self.rotate[2].to_radians());
        // Z-last means Z outermost: rotate about X, then Y, then Z, with the
        // scale innermost of all.
        let r = rz.compose(&ry.compose(&rx.compose(&Xform::scale(self.scale))));
        Xform::translate(v3(self.at[0], self.at[1], self.at[2])).compose(&r)
    }
}

/// Reads the file and returns the placements, with `file` entries resolved
/// relative to the scene file's own directory.
pub fn parse(path: &Path) -> Result<Vec<PlaceSpec>, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let dir = path.parent().unwrap_or(Path::new("."));
    let mut out: Vec<PlaceSpec> = Vec::new();
    let mut open = false;

    for (n, raw) in text.lines().enumerate() {
        let n = n + 1;
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        if line == "[[place]]" {
            if open && out.last().is_some_and(|p| p.file.as_os_str().is_empty()) {
                return Err(format!("line {n}: the previous [[place]] never named a file"));
            }
            out.push(PlaceSpec::default());
            open = true;
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("line {n}: expected `key = value` or [[place]], got {line:?}"));
        };
        if !open {
            return Err(format!("line {n}: {key:?} before the first [[place]]"));
        }
        let (key, value) = (key.trim(), value.trim());
        let place = out.last_mut().unwrap();
        match key {
            "file" => {
                let name = value
                    .strip_prefix('"')
                    .and_then(|v| v.strip_suffix('"'))
                    .ok_or_else(|| format!("line {n}: file wants a quoted string"))?;
                place.file = dir.join(name);
            }
            "at" => place.at = numbers::<3>(value).map_err(|e| format!("line {n}: at {e}"))?,
            "rotate" => {
                place.rotate = numbers::<3>(value).map_err(|e| format!("line {n}: rotate {e}"))?
            }
            "scale" => {
                place.scale = value
                    .parse()
                    .map_err(|_| format!("line {n}: scale {value:?} is not a number"))?;
                if !(place.scale > 0.0) {
                    return Err(format!("line {n}: scale must be positive"));
                }
            }
            "tint" => {
                let t = numbers::<4>(value).map_err(|e| format!("line {n}: tint {e}"))?;
                place.tint = Some(t.map(|x| x as f32));
            }
            other => return Err(format!("line {n}: unknown key {other:?}")),
        }
    }
    if out.is_empty() {
        return Err(format!("{}: no [[place]] blocks", path.display()));
    }
    if let Some(p) = out.iter().find(|p| p.file.as_os_str().is_empty()) {
        let _ = p;
        return Err("a [[place]] block never named a file".into());
    }
    Ok(out)
}

/// `[a, b, c]` with exactly N numbers.
fn numbers<const N: usize>(s: &str) -> Result<[f64; N], String> {
    let inner = s
        .strip_prefix('[')
        .and_then(|v| v.strip_suffix(']'))
        .ok_or_else(|| format!("wants [{N} numbers]"))?;
    let mut out = [0.0; N];
    let mut count = 0;
    for (i, part) in inner.split(',').enumerate() {
        if i >= N {
            return Err(format!("wants exactly {N} numbers"));
        }
        out[i] = part
            .trim()
            .parse()
            .map_err(|_| format!("{:?} is not a number", part.trim()))?;
        count = i + 1;
    }
    if count != N {
        return Err(format!("wants exactly {N} numbers, got {count}"));
    }
    Ok(out)
}

/// Imports every placed file (once each, via the [`Assets`] cache) and
/// builds the scene. A failure names the placement that wanted the file.
pub fn build(specs: &[PlaceSpec], assets: &Assets) -> Result<Scene, String> {
    let mut scene = Scene::new();
    for (i, spec) in specs.iter().enumerate() {
        let mesh = assets
            .import(&spec.file)
            .map_err(|e| format!("place {} of {}: {e}", i + 1, specs.len()))?;
        scene.add_tinted(mesh, spec.xform(), spec.tint);
    }
    Ok(scene)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write(name: &str, text: &str) -> PathBuf {
        let dir = std::env::temp_dir().join("stpview-scene-tests");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(name);
        std::fs::write(&p, text).unwrap();
        p
    }

    #[test]
    fn a_full_scene_parses_and_paths_are_relative_to_the_file() {
        let p = write(
            "ok.toml",
            "# demo\n\
             [[place]]\n\
             file = \"board.step\"\n\
             \n\
             [[place]]\n\
             file = \"pin.step\"  # a comment\n\
             at = [10.0, 5, 1.6]\n\
             rotate = [0, 0, 90]\n\
             tint = [0.9, 0.9, 0.9, 1.0]\n",
        );
        let specs = parse(&p).unwrap();
        assert_eq!(specs.len(), 2);
        assert!(specs[0].file.ends_with("board.step"));
        assert!(specs[0].file.parent().unwrap().ends_with("stpview-scene-tests"));
        assert_eq!(specs[1].at, [10.0, 5.0, 1.6]);
        assert_eq!(specs[1].rotate, [0.0, 0.0, 90.0]);
        assert_eq!(specs[1].tint, Some([0.9, 0.9, 0.9, 1.0]));
    }

    /// Errors carry the line number of a hand-edited scene file.
    #[test]
    fn errors_name_their_line() {
        for (text, want) in [
            ("file = \"a.step\"\n", "line 1"),
            ("[[place]]\nfile = bare\n", "line 2"),
            ("[[place]]\nfile = \"a.step\"\nat = [1, 2]\n", "line 3"),
            ("[[place]]\nfile = \"a.step\"\ncolour = [1]\n", "line 3"),
        ] {
            let p = write("bad.toml", text);
            let e = parse(&p).unwrap_err();
            assert!(e.contains(want), "{text:?} -> {e}");
        }
    }

    /// The rotation order is the documented one: X, then Y, then Z. A point
    /// on +X rotated 90 about X stays put, then 90 about Z carries it to +Y.
    #[test]
    fn rotation_order_is_x_then_y_then_z() {
        let spec = PlaceSpec {
            file: PathBuf::from("x"),
            rotate: [90.0, 0.0, 90.0],
            ..PlaceSpec::default()
        };
        let p = spec.xform().point(v3(1.0, 0.0, 0.0));
        assert!(p.distance(v3(0.0, 1.0, 0.0)) < 1e-12, "{p:?}");
    }
}
