//! Selecting parts by clicking on them, and walking the assembly they belong to.
//!
//! There is no tree widget (no UI toolkit), so the assembly is walked from the
//! picture: a click selects the body under the cursor, and Up widens that to
//! its component, subassembly, and the whole model. The selection is painted
//! and named in the title bar; hiding, isolating and fitting act on it.

use stpview::model::AsmNode;
use stpview::render::Paint;
use stpview::tess::Part;

/// The assembly tree as selection needs it: a name and a parent per node, in
/// `Model::nodes` order.
#[derive(Default, Clone)]
pub struct Hierarchy {
    names: Vec<String>,
    parents: Vec<Option<usize>>,
    /// How many named nodes have no named node above them; decides what "top
    /// level" means when colouring by assembly.
    roots: usize,
}

impl Hierarchy {
    pub fn new(nodes: &[AsmNode]) -> Hierarchy {
        let mut h = Hierarchy {
            names: nodes.iter().map(|n| n.name.clone()).collect(),
            parents: nodes.iter().map(|n| n.parent).collect(),
            roots: 0,
        };
        h.roots = (0..nodes.len())
            .filter(|&i| !h.names[i].is_empty() && h.named_ancestry(i).len() == 1)
            .count();
        h
    }

    /// `node` and everything above it, nearest first. A node this tree does
    /// not know (a scene's parts index their own models) has no ancestry.
    fn ancestry(&self, node: usize) -> impl Iterator<Item = usize> + '_ {
        std::iter::successors(Some(node).filter(|&n| n < self.names.len()), move |&n| {
            self.parents.get(n).copied().flatten()
        })
    }

    /// The named nodes at and above a node, nearest first. Unnamed nodes are
    /// skipped: a component's geometry hangs off a nameless representation
    /// below its product, which `--tree` folds away too.
    fn named_ancestry(&self, node: usize) -> Vec<usize> {
        self.ancestry(node)
            .filter(|&n| !self.names[n].is_empty())
            .collect()
    }

    /// The path down the assembly to a node, as `--tree` prints it.
    pub fn path(&self, node: usize) -> String {
        let mut names: Vec<&str> = self
            .named_ancestry(node)
            .into_iter()
            .map(|n| self.names[n].as_str())
            .collect();
        names.reverse();
        names.join(" / ")
    }

    /// Every part whose node lies under `node`, or is it.
    fn parts_under(&self, node: usize, part_nodes: &[Option<usize>]) -> Vec<usize> {
        part_nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.is_some_and(|n| self.ancestry(n).any(|a| a == node)))
            .map(|(i, _)| i)
            .collect()
    }

    /// The top-level component a part's node belongs to, for colouring by
    /// assembly: the named node one below the single root, or the root
    /// itself when there are several. `None` where there is no such level (no
    /// product structure, or a body directly under the root).
    fn top_component(&self, node: usize) -> Option<usize> {
        let chain = self.named_ancestry(node);
        match chain.len() {
            0 => None,
            1 if self.roots > 1 => Some(chain[0]),
            1 => None,
            n if self.roots > 1 => Some(chain[n - 1]),
            n => Some(chain[n - 2]),
        }
    }
}

/// What is selected: a set of parts, and how it was arrived at, since Up and
/// Down walk on from where the last click landed.
#[derive(Default, Clone, Debug)]
pub struct Selection {
    /// One flag per part; missing trailing flags are unselected, so it can be
    /// built before a streamed load is complete.
    flags: Vec<bool>,
    /// The part the last click landed on.
    anchor: Option<usize>,
    /// The node the selection has been widened to, when it has been: every
    /// part under it is selected, and it is what the readout names.
    level: Option<usize>,
}

impl Selection {
    pub fn is_empty(&self) -> bool {
        !self.flags.iter().any(|&f| f)
    }

    pub fn count(&self) -> usize {
        self.flags.iter().filter(|&&f| f).count()
    }

    pub fn contains(&self, part: usize) -> bool {
        self.flags.get(part).copied().unwrap_or(false)
    }

    /// The selected parts, in buffer order.
    pub fn parts(&self) -> impl Iterator<Item = usize> + '_ {
        self.flags
            .iter()
            .enumerate()
            .filter(|(_, f)| **f)
            .map(|(i, _)| i)
    }

    pub fn clear(&mut self) {
        *self = Selection::default();
    }

    /// A click: this part and nothing else.
    pub fn set(&mut self, part: usize) {
        self.flags.clear();
        self.flags.resize(part + 1, false);
        self.flags[part] = true;
        self.anchor = Some(part);
        self.level = None;
    }

    /// A ctrl-click: the part joins the selection, or leaves it again.
    pub fn toggle(&mut self, part: usize) {
        if self.flags.len() <= part {
            self.flags.resize(part + 1, false);
        }
        self.flags[part] = !self.flags[part];
        self.anchor = if self.flags[part] {
            Some(part)
        } else {
            self.parts().last()
        };
        self.level = None;
    }

    fn assign(&mut self, parts: &[usize], level: usize) {
        let n = parts.iter().map(|&p| p + 1).max().unwrap_or(0);
        self.flags = vec![false; n];
        for &p in parts {
            self.flags[p] = true;
        }
        self.level = Some(level);
    }

    /// Up: every part under the next named node above the clicked part that
    /// holds more parts than are selected now. Says whether there was a level
    /// to go to.
    pub fn widen(&mut self, h: &Hierarchy, part_nodes: &[Option<usize>]) -> bool {
        let Some(anchor) = self.anchor else { return false };
        let Some(node) = part_nodes.get(anchor).copied().flatten() else { return false };
        let chain = h.named_ancestry(node);
        let from = match self.level {
            Some(l) => chain.iter().position(|&n| n == l).map_or(0, |i| i + 1),
            None => 0,
        };
        let have = self.count();
        for &n in &chain[from..] {
            let parts = h.parts_under(n, part_nodes);
            if parts.len() > have {
                self.assign(&parts, n);
                return true;
            }
        }
        false
    }

    /// Down: back to the level below the current one -- the first with fewer
    /// parts -- or to the clicked part alone. Says whether anything changed.
    pub fn narrow(&mut self, h: &Hierarchy, part_nodes: &[Option<usize>]) -> bool {
        let (Some(anchor), Some(level)) = (self.anchor, self.level) else { return false };
        let Some(node) = part_nodes.get(anchor).copied().flatten() else { return false };
        let chain = h.named_ancestry(node);
        let at = chain.iter().position(|&n| n == level).unwrap_or(0);
        let have = self.count();
        for &n in chain[..at].iter().rev() {
            let parts = h.parts_under(n, part_nodes);
            // A level holding only the clicked body is that body; Up
            // skipped it too.
            if parts.len() < have && parts.len() > 1 {
                self.assign(&parts, n);
                return true;
            }
        }
        self.set(anchor);
        true
    }

    /// What to call it, for the title bar and the console.
    pub fn describe(&self, h: &Hierarchy, parts: &[Part]) -> String {
        let n = self.count();
        if n == 0 {
            return String::new();
        }
        if let Some(level) = self.level {
            let s = if n == 1 { "" } else { "s" };
            return format!("{} ({n} part{s})", h.path(level));
        }
        if n == 1 {
            return part_label(parts, self.parts().next().unwrap_or(0));
        }
        format!("{n} parts")
    }
}

/// A part's name, or its number when the file gave it none.
pub fn part_label(parts: &[Part], i: usize) -> String {
    match parts.get(i) {
        Some(p) if !p.name.is_empty() => p.name.clone(),
        _ => format!("part {} of {}", i + 1, parts.len()),
    }
}

/// How the parts are coloured: as the file says, or to tell them apart.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum ColorMode {
    /// The colours the file gives them, which is often one grey.
    #[default]
    File,
    /// Each part a hue of its own.
    Part,
    /// Each top-level component a hue of its own, so a subassembly reads as
    /// one thing.
    Assembly,
}

impl ColorMode {
    pub fn next(self) -> ColorMode {
        match self {
            ColorMode::File => ColorMode::Part,
            ColorMode::Part => ColorMode::Assembly,
            ColorMode::Assembly => ColorMode::File,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ColorMode::File => "as the file has them",
            ColorMode::Part => "by part",
            ColorMode::Assembly => "by assembly",
        }
    }

    pub fn parse(s: &str) -> Option<ColorMode> {
        match s.to_ascii_lowercase().as_str() {
            "file" => Some(ColorMode::File),
            "part" | "parts" => Some(ColorMode::Part),
            "assembly" | "assemblies" => Some(ColorMode::Assembly),
            _ => None,
        }
    }
}

/// The selection accent (the icon's amber), for parts that are not warm.
const AMBER: [f32; 3] = [1.0, 0.479, 0.033];

/// The accent for red, orange and yellow parts, on which amber barely shows.
const BLUE: [f32; 3] = [0.25, 0.55, 1.0];

/// How far a selected part goes towards its accent: enough to read on black
/// and white, leaving some of its own colour visible.
pub const SELECTED_AMOUNT: f32 = 0.78;

/// The part under the cursor before it is clicked: the same accent, faintly.
pub const HOVERED_AMOUNT: f32 = 0.30;

/// Whether a colour is red, orange or yellow: red its strongest channel, and
/// well clear of blue. Greys and whites are not, and take amber.
fn warm(c: [f32; 3]) -> bool {
    c[0] >= c[1] && c[0] - c[2] > 0.3
}

/// The highlight for a part of this colour, at `amount`: the accent that
/// contrasts with it.
pub fn highlight(color: [f32; 3], amount: f32) -> Paint {
    Paint {
        rgb: if warm(color) { BLUE } else { AMBER },
        amount,
    }
}

/// Each part's node in the assembly, which is what a selection walks.
pub fn nodes(parts: &[Part]) -> Vec<Option<usize>> {
    parts.iter().map(|p| p.node).collect()
}

/// Twelve hues to tell parts apart by, in linear light. None is near the
/// selection amber, and neighbours in the list (often neighbours in the
/// model) are far apart on the wheel.
const PALETTE: [[f32; 3]; 12] = [
    [0.15, 0.35, 0.95], // blue
    [0.15, 0.65, 0.20], // green
    [0.90, 0.12, 0.12], // red
    [0.55, 0.20, 0.90], // violet
    [0.08, 0.60, 0.60], // teal
    [0.90, 0.15, 0.55], // magenta
    [0.50, 0.85, 0.12], // lime
    [0.25, 0.65, 0.98], // sky
    [0.95, 0.45, 0.55], // rose
    [0.22, 0.18, 0.80], // indigo
    [0.20, 0.85, 0.55], // mint
    [0.70, 0.25, 0.60], // plum
];

/// How far towards its hue a part is taken: all the way, so one hue looks the
/// same on every part. Paint replaces the colour before lighting, so shading
/// remains.
const HUE_AMOUNT: f32 = 1.0;

/// Which hue each part gets in `mode`: its own, or its top-level component's,
/// numbered in order of first appearance. `None` is unpainted.
fn hues(mode: ColorMode, h: &Hierarchy, part_nodes: &[Option<usize>]) -> Vec<Option<usize>> {
    match mode {
        ColorMode::File => vec![None; part_nodes.len()],
        ColorMode::Part => (0..part_nodes.len()).map(Some).collect(),
        ColorMode::Assembly => {
            let mut seen: Vec<(usize, usize)> = Vec::new();
            let mut next = 0usize;
            let mut fresh = || {
                next += 1;
                next - 1
            };
            part_nodes
                .iter()
                .map(|n| match n.and_then(|n| h.top_component(n)) {
                    Some(k) => match seen.iter().find(|(key, _)| *key == k) {
                        Some(&(_, hue)) => Some(hue),
                        None => {
                            let hue = fresh();
                            seen.push((k, hue));
                            Some(hue)
                        }
                    },
                    None => Some(fresh()),
                })
                .collect()
        }
    }
}

/// The paint for every part: the colour mode underneath, the part under the
/// cursor over that, and the selection over everything. Empty (the renderer's
/// fast path) when there is nothing to paint.
pub fn paint(
    mode: ColorMode,
    h: &Hierarchy,
    parts: &[Part],
    selection: &Selection,
    hover: Option<usize>,
) -> Vec<Paint> {
    let n = parts.len();
    if mode == ColorMode::File && selection.is_empty() && hover.is_none() {
        return Vec::new();
    }
    let mut out: Vec<Paint> = hues(mode, h, &nodes(parts))
        .into_iter()
        .map(|hue| match hue {
            Some(k) => Paint {
                rgb: PALETTE[k % PALETTE.len()],
                amount: HUE_AMOUNT,
            },
            None => Paint::NONE,
        })
        .collect();
    if let Some(hv) = hover
        && hv < n
        && !selection.contains(hv)
    {
        out[hv] = highlight(parts[hv].color, HOVERED_AMOUNT);
    }
    for i in selection.parts() {
        if i < n {
            out[i] = highlight(parts[i].color, SELECTED_AMOUNT);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use stpview::math::Aabb;

    fn node(name: &str, parent: Option<usize>) -> AsmNode {
        AsmNode {
            name: name.into(),
            parent,
        }
    }

    fn part(name: &str, node: Option<usize>) -> Part {
        Part {
            name: name.into(),
            node,
            alpha: 1.0,
            cappable: true,
            color: [0.5; 3],
            tris: 0..0,
            verts: 0..0,
            wire: 0..0,
            sharp: 0..0,
            tangent: 0..0,
            bbox: Aabb::EMPTY,
            faces: Vec::new(),
        }
    }

    /// A plate and a rod assembly with a rod and two nuts, each product with
    /// the nameless representation node its geometry hangs off, plus one
    /// body outside the assembly.
    fn as1() -> (Hierarchy, Vec<Option<usize>>) {
        let nodes = vec![
            node("AS1", None),        // 0
            node("PLATE", Some(0)),   // 1
            node("", Some(1)),        // 2  the plate's body
            node("ROD-ASM", Some(0)), // 3
            node("ROD", Some(3)),     // 4
            node("", Some(4)),        // 5
            node("NUT", Some(3)),     // 6
            node("", Some(6)),        // 7
            node("NUT", Some(3)),     // 8  the second occurrence
            node("", Some(8)),        // 9
        ];
        let parts = vec![Some(2), Some(5), Some(7), Some(9), None];
        (Hierarchy::new(&nodes), parts)
    }

    fn selected(s: &Selection) -> Vec<usize> {
        s.parts().collect()
    }

    /// Up goes to the first level that adds something, and Down comes back
    /// through the same levels, so a one-body component is never a stop.
    #[test]
    fn up_walks_to_the_subassembly_and_the_model_and_down_again() {
        let (h, parts) = as1();
        let mut s = Selection::default();
        s.set(2);
        assert_eq!(selected(&s), vec![2]);
        // NUT holds one body, the one selected, so Up goes straight past it.
        assert!(s.widen(&h, &parts));
        assert_eq!(selected(&s), vec![1, 2, 3]);
        assert_eq!(s.level, Some(3));
        assert!(s.widen(&h, &parts));
        // The loose body is outside the assembly, so the root is four parts.
        assert_eq!(selected(&s), vec![0, 1, 2, 3]);
        assert!(!s.widen(&h, &parts), "there is nothing above the root");
        assert_eq!(selected(&s), vec![0, 1, 2, 3]);

        assert!(s.narrow(&h, &parts));
        assert_eq!(selected(&s), vec![1, 2, 3]);
        assert!(s.narrow(&h, &parts));
        assert_eq!(selected(&s), vec![2]);
        assert_eq!(s.level, None);
        assert!(!s.narrow(&h, &parts));
    }

    #[test]
    fn a_body_outside_the_assembly_has_nothing_to_widen_to() {
        let (h, parts) = as1();
        let mut s = Selection::default();
        s.set(4);
        assert!(!s.widen(&h, &parts));
        assert_eq!(selected(&s), vec![4]);
    }

    #[test]
    fn a_ctrl_click_adds_and_removes_and_the_anchor_follows() {
        let (h, parts) = as1();
        let mut s = Selection::default();
        s.set(0);
        s.toggle(2);
        assert_eq!(selected(&s), vec![0, 2]);
        // Up walks from the last part clicked, which is the nut.
        assert!(s.widen(&h, &parts));
        assert_eq!(selected(&s), vec![1, 2, 3]);
        s.toggle(2);
        assert_eq!(selected(&s), vec![1, 3]);
        s.toggle(1);
        s.toggle(3);
        assert!(s.is_empty());
    }

    #[test]
    fn the_readout_names_the_part_the_level_or_the_count() {
        let (h, parts) = as1();
        let names = ["AS1 / PLATE", "AS1 / ROD-ASM / ROD", "AS1 / ROD-ASM / NUT", "AS1 / ROD-ASM / NUT", ""];
        let ps: Vec<Part> = names.iter().zip(&parts).map(|(n, node)| part(n, *node)).collect();
        let mut s = Selection::default();
        assert_eq!(s.describe(&h, &ps), "");
        s.set(1);
        assert_eq!(s.describe(&h, &ps), "AS1 / ROD-ASM / ROD");
        s.widen(&h, &parts);
        assert_eq!(s.describe(&h, &ps), "AS1 / ROD-ASM (3 parts)");
        s.set(4);
        assert_eq!(s.describe(&h, &ps), "part 5 of 5");
        s.toggle(0);
        assert_eq!(s.describe(&h, &ps), "2 parts");
    }

    #[test]
    fn colouring_by_assembly_gives_a_subassembly_one_hue() {
        let (h, parts) = as1();
        assert_eq!(
            hues(ColorMode::Assembly, &h, &parts),
            vec![Some(0), Some(1), Some(1), Some(1), Some(2)]
        );
        assert_eq!(
            hues(ColorMode::Part, &h, &parts),
            vec![Some(0), Some(1), Some(2), Some(3), Some(4)]
        );
        assert!(hues(ColorMode::File, &h, &parts).iter().all(|h| h.is_none()));
    }

    /// One product with many bodies has no level between a body and the root,
    /// so "by assembly" falls back to a hue per part.
    #[test]
    fn a_flat_file_colours_by_part_whichever_mode_is_asked() {
        let nodes = vec![node("Top", None), node("", Some(0))];
        let parts = vec![Some(1), Some(1), Some(1)];
        let h = Hierarchy::new(&nodes);
        assert_eq!(
            hues(ColorMode::Assembly, &h, &parts),
            vec![Some(0), Some(1), Some(2)]
        );
    }

    #[test]
    fn several_roots_are_themselves_the_top_level() {
        let nodes = vec![node("A", None), node("", Some(0)), node("B", None), node("", Some(2))];
        let parts = vec![Some(1), Some(1), Some(3)];
        let h = Hierarchy::new(&nodes);
        assert_eq!(hues(ColorMode::Assembly, &h, &parts), vec![Some(0), Some(0), Some(1)]);
    }

    #[test]
    fn the_selection_is_painted_over_the_hues_and_the_hover_under_it() {
        let (h, nodes) = as1();
        // A red plate, a blue rod, teal nuts and a grey stray.
        let colors = [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0], [0.5; 3]];
        let parts: Vec<Part> = nodes
            .iter()
            .zip(colors)
            .map(|(n, c)| {
                let mut p = part("", *n);
                p.color = c;
                p
            })
            .collect();
        let mut s = Selection::default();
        assert!(
            paint(ColorMode::File, &h, &parts, &s, None).is_empty(),
            "nothing to paint is the renderer's fast path"
        );
        s.set(1);
        let p = paint(ColorMode::Assembly, &h, &parts, &s, Some(2));
        assert_eq!(p.len(), 5);
        assert_eq!(p[1], Paint { rgb: AMBER, amount: SELECTED_AMOUNT });
        assert_eq!(p[2], Paint { rgb: AMBER, amount: HOVERED_AMOUNT });
        assert_eq!(p[0].amount, HUE_AMOUNT);
        // Hovering a selected part changes nothing about it.
        let p = paint(ColorMode::File, &h, &parts, &s, Some(1));
        assert_eq!(p[1].amount, SELECTED_AMOUNT);
        assert_eq!(p[0], Paint::NONE);
        // The red plate takes the cool accent, since amber on red is red.
        s.set(0);
        assert_eq!(paint(ColorMode::File, &h, &parts, &s, None)[0].rgb, BLUE);
        // A hover past the end of the parts paints nothing.
        assert_eq!(paint(ColorMode::File, &h, &parts, &s, Some(99)).len(), 5);
    }

    /// Warm parts get the blue accent; everything else, greys and whites
    /// included, gets amber.
    #[test]
    fn the_accent_is_the_one_the_part_is_not() {
        for warm in [[1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.5, 0.0], [0.6, 0.3, 0.1]] {
            assert_eq!(highlight(warm, 1.0).rgb, BLUE, "{warm:?}");
        }
        for cool in [[0.5; 3], [1.0; 3], [0.0; 3], [0.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 1.0]] {
            assert_eq!(highlight(cool, 1.0).rgb, AMBER, "{cool:?}");
        }
        assert_eq!(highlight([0.5; 3], HOVERED_AMOUNT).amount, HOVERED_AMOUNT);
    }

    /// A node index the tree does not know (as in a scene) is a loose body,
    /// not a panic.
    #[test]
    fn a_node_the_hierarchy_does_not_know_is_loose() {
        let h = Hierarchy::default();
        let parts = vec![Some(7), Some(3)];
        let mut s = Selection::default();
        s.set(0);
        assert!(!s.widen(&h, &parts));
        assert_eq!(hues(ColorMode::Assembly, &h, &parts), vec![Some(0), Some(1)]);
        assert_eq!(h.path(7), "");
    }
}
