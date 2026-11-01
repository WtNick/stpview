//! ISO-10303-21 ("STEP physical file") parser.
//!
//! The parser is arena-based: every parsed value lands in one flat `Vec<Value>`
//! and every string in one flat byte buffer, avoiding individually boxed nodes
//! on files with millions of entities. `Value` is deliberately kept at 16
//! bytes.

use std::collections::HashMap;
use std::fmt;

/// Marker used in `Record::type_id` for ids that never appear in the file.
pub const TYPE_ABSENT: u16 = u16::MAX;

#[derive(Clone, Copy, Debug)]
pub enum Value {
    /// `$` -- unset optional attribute.
    Null,
    /// `*` -- value derived in a subtype.
    Star,
    /// `#123`
    Ref(u32),
    Int(i64),
    Real(f64),
    /// `'text'`, stored as a range into the string arena.
    Str { off: u32, len: u32 },
    /// `.ENUM.`, stored without the dots.
    Enum { off: u32, len: u32 },
    /// `(a, b, c)` -- range into the value arena.
    List { start: u32, len: u32 },
    /// `NAME(...)` appearing inside a complex instance; indexes `typed`.
    Typed(u32),
}

/// One `NAME(...)` group of a complex instance.
#[derive(Clone, Copy, Debug)]
pub struct TypedEntry {
    pub type_id: u16,
    pub start: u32,
    pub len: u32,
}

#[derive(Clone, Copy, Debug)]
pub struct Record {
    pub type_id: u16,
    pub start: u32,
    pub len: u32,
}

impl Record {
    pub const ABSENT: Record = Record {
        type_id: TYPE_ABSENT,
        start: 0,
        len: 0,
    };

    pub fn exists(&self) -> bool {
        self.type_id != TYPE_ABSENT
    }
}

pub struct StepFile {
    records: Vec<Record>,
    values: Vec<Value>,
    typed: Vec<TypedEntry>,
    strings: Vec<u8>,
    type_names: Vec<String>,
    /// Number of entity instances actually present.
    pub entity_count: usize,
}

impl StepFile {
    #[inline]
    pub fn record(&self, id: u32) -> Record {
        self.records
            .get(id as usize)
            .copied()
            .unwrap_or(Record::ABSENT)
    }

    #[inline]
    pub fn type_name(&self, type_id: u16) -> &str {
        self.type_names
            .get(type_id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    /// Type name of the entity `id`, or `""` if that id is not in the file.
    #[inline]
    pub fn type_of(&self, id: u32) -> &str {
        self.type_name(self.record(id).type_id)
    }

    /// Attributes of entity `id`.
    #[inline]
    pub fn params(&self, id: u32) -> &[Value] {
        let r = self.record(id);
        self.slice(r.start, r.len)
    }

    #[inline]
    pub fn slice(&self, start: u32, len: u32) -> &[Value] {
        &self.values[start as usize..(start + len) as usize]
    }

    /// Elements of a `Value::List`.
    #[inline]
    pub fn list(&self, v: Value) -> &[Value] {
        match v {
            Value::List { start, len } => self.slice(start, len),
            _ => &[],
        }
    }

    pub fn typed_entry(&self, idx: u32) -> TypedEntry {
        self.typed[idx as usize]
    }

    /// The arguments of a typed value such as `PLANE_ANGLE_MEASURE(0.017)`,
    /// or empty if `v` is not one.
    pub fn typed_params(&self, v: Value) -> &[Value] {
        match v {
            Value::Typed(idx) => {
                let t = self.typed_entry(idx);
                self.slice(t.start, t.len)
            }
            _ => &[],
        }
    }

    pub fn str_of(&self, v: Value) -> &str {
        match v {
            Value::Str { off, len } | Value::Enum { off, len } => {
                std::str::from_utf8(&self.strings[off as usize..(off + len) as usize])
                    .unwrap_or("")
            }
            _ => "",
        }
    }

    /// All entity ids in the file, ascending.
    pub fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        self.records
            .iter()
            .enumerate()
            .filter(|(_, r)| r.exists())
            .map(|(i, _)| i as u32)
    }

    /// Every entity of one simple type, ascending. The name is looked up once,
    /// so a pass over a million records is an integer compare apiece.
    pub fn ids_of<'a>(&'a self, name: &str) -> impl Iterator<Item = u32> + 'a {
        let want = self.type_names.iter().position(|t| t == name).map(|i| i as u16);
        self.records
            .iter()
            .enumerate()
            .filter(move |(_, r)| want == Some(r.type_id))
            .map(|(i, _)| i as u32)
    }

    /// Highest entity id + 1.
    pub fn id_bound(&self) -> u32 {
        self.records.len() as u32
    }

    /// For a complex instance, the parameters of its `name(...)` group.
    /// Returns `None` if this entity is not complex or lacks that group.
    pub fn complex_part(&self, id: u32, name: &str) -> Option<&[Value]> {
        let r = self.record(id);
        if self.type_name(r.type_id) != COMPLEX_TYPE {
            return None;
        }
        for v in self.slice(r.start, r.len) {
            if let Value::Typed(idx) = *v {
                let t = self.typed_entry(idx);
                if self.type_name(t.type_id) == name {
                    return Some(self.slice(t.start, t.len));
                }
            }
        }
        None
    }
}

/// Synthetic type name given to complex instances `#N=(A(..)B(..));`.
pub const COMPLEX_TYPE: &str = "$COMPLEX";

/// ISO 10303-21 permits entity names in an abbreviated form, and some
/// translators use it throughout. Names are expanded as they are interned, so
/// nothing downstream knows the file was written that way.
///
/// The table is deliberately partial; an unlisted name passes through and is
/// reported as unsupported. Most entries come from the standard's list: a
/// wrong one costs nothing, a missing one costs the entity.
/// `SURFACE_OF_LINEAR_EXTRUSION` has two entries: `SRFLEX` from the list, and
/// `SL`, which translators are seen to write.
fn expand_short_name(name: &[u8]) -> Option<&'static str> {
    Some(match name {
        // Geometry.
        b"CRTPNT" => "CARTESIAN_POINT",
        b"DRCTN" => "DIRECTION",
        b"A2PL3D" => "AXIS2_PLACEMENT_3D",
        b"AX1PLC" => "AXIS1_PLACEMENT",
        b"CYLSRF" => "CYLINDRICAL_SURFACE",
        b"CNCSRF" => "CONICAL_SURFACE",
        b"SPHSRF" => "SPHERICAL_SURFACE",
        b"TORSRF" => "TOROIDAL_SURFACE",
        b"SRFRV" => "SURFACE_OF_REVOLUTION",
        b"SRFLEX" | b"SL" => "SURFACE_OF_LINEAR_EXTRUSION",
        b"RCTRSR" => "RECTANGULAR_TRIMMED_SURFACE",
        b"BSCWK" => "B_SPLINE_CURVE_WITH_KNOTS",
        b"BSSWK" => "B_SPLINE_SURFACE_WITH_KNOTS",
        b"ELLPS" => "ELLIPSE",
        b"TRMCRV" => "TRIMMED_CURVE",
        b"CMPCRV" => "COMPOSITE_CURVE",
        b"CMCRSG" => "COMPOSITE_CURVE_SEGMENT",
        b"SRFCRV" => "SURFACE_CURVE",
        b"INTCR" => "INTERSECTION_CURVE",
        b"SMCRV" => "SEAM_CURVE",
        b"GMTST" => "GEOMETRIC_SET",
        // Topology.
        b"VRTPNT" => "VERTEX_POINT",
        b"EDGCRV" => "EDGE_CURVE",
        b"EDGLP" => "EDGE_LOOP",
        b"ORNEDG" => "ORIENTED_EDGE",
        b"PLYLP" => "POLY_LOOP",
        b"FCBND" => "FACE_BOUND",
        b"FCOTBN" => "FACE_OUTER_BOUND",
        b"FCSRF" => "FACE_SURFACE",
        b"ADVFC" => "ADVANCED_FACE",
        b"CLSSHL" => "CLOSED_SHELL",
        b"OPNSHL" => "OPEN_SHELL",
        b"ORCLSH" => "ORIENTED_CLOSED_SHELL",
        b"CNFCST" => "CONNECTED_FACE_SET",
        b"MNSDBRP" => "MANIFOLD_SOLID_BREP",
        b"BRWTVD" => "BREP_WITH_VOIDS",
        b"FCTBR" => "FACETED_BREP",
        b"SBSM" => "SHELL_BASED_SURFACE_MODEL",
        // Representations.
        b"SHPRPR" => "SHAPE_REPRESENTATION",
        b"ADBSR" => "ADVANCED_BREP_SHAPE_REPRESENTATION",
        b"FBSR" => "FACETED_BREP_SHAPE_REPRESENTATION",
        b"GBSSR" => "GEOMETRICALLY_BOUNDED_SURFACE_SHAPE_REPRESENTATION",
        b"MSSHRP" => "MANIFOLD_SURFACE_SHAPE_REPRESENTATION",
        b"SHDFRP" => "SHAPE_DEFINITION_REPRESENTATION",
        b"SHRPRL" => "SHAPE_REPRESENTATION_RELATIONSHIP",
        // Units, and the context that assigns them. These travel together in
        // one complex instance; without them the length unit is never found
        // and an inch file silently passes for millimetres.
        b"LNGUNT" => "LENGTH_UNIT",
        b"PLANUN" => "PLANE_ANGLE_UNIT",
        b"NMDUNT" => "NAMED_UNIT",
        b"SUNT" => "SI_UNIT",
        b"CNBSUN" => "CONVERSION_BASED_UNIT",
        b"LMWU" => "LENGTH_MEASURE_WITH_UNIT",
        b"PAMWU" => "PLANE_ANGLE_MEASURE_WITH_UNIT",
        b"UMWU" => "UNCERTAINTY_MEASURE_WITH_UNIT",
        b"GMRPCN" => "GEOMETRIC_REPRESENTATION_CONTEXT",
        b"GUAC" => "GLOBAL_UNIT_ASSIGNED_CONTEXT",
        b"GC" => "GLOBAL_UNCERTAINTY_ASSIGNED_CONTEXT",
        b"RPRCNT" => "REPRESENTATION_CONTEXT",
        _ => return None,
    })
}

#[derive(Debug)]
pub struct ParseError {
    pub offset: usize,
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "STEP parse error at line {} (byte {}): {}",
            self.line, self.offset, self.message
        )
    }
}

impl std::error::Error for ParseError {}

struct Parser<'a> {
    src: &'a [u8],
    pos: usize,
    values: Vec<Value>,
    typed: Vec<TypedEntry>,
    strings: Vec<u8>,
    type_names: Vec<String>,
    type_ids: HashMap<Box<[u8]>, u16>,
    /// Scratch stack shared by all list/parameter parsing, so nested lists do
    /// not each allocate a temporary Vec.
    scratch: Vec<Value>,
}

impl<'a> Parser<'a> {
    fn new(src: &'a [u8]) -> Self {
        // Empirically a STEP file yields roughly one arena value per 18 bytes.
        let est = src.len() / 18 + 64;
        let mut p = Parser {
            src,
            pos: 0,
            values: Vec::with_capacity(est),
            typed: Vec::new(),
            strings: Vec::with_capacity(src.len() / 200 + 64),
            type_names: Vec::new(),
            type_ids: HashMap::new(),
            scratch: Vec::with_capacity(256),
        };
        p.intern(COMPLEX_TYPE.as_bytes());
        p
    }

    fn intern(&mut self, name: &[u8]) -> u16 {
        // Abbreviated names become their full form here, so the rest of the
        // reader never sees them. The inequality guard matters: an entry
        // mapping a name to itself would recurse forever.
        if let Some(full) = expand_short_name(name).filter(|f| f.as_bytes() != name) {
            return self.intern(full.as_bytes());
        }
        if let Some(&id) = self.type_ids.get(name) {
            return id;
        }
        let id = self.type_names.len() as u16;
        assert!(id != TYPE_ABSENT, "too many distinct entity types");
        self.type_names
            .push(String::from_utf8_lossy(name).into_owned());
        self.type_ids.insert(name.into(), id);
        id
    }

    fn err(&self, message: impl Into<String>) -> ParseError {
        let line = 1 + self.src[..self.pos.min(self.src.len())]
            .iter()
            .filter(|&&b| b == b'\n')
            .count();
        ParseError {
            offset: self.pos,
            line,
            message: message.into(),
        }
    }

    #[inline]
    fn peek(&self) -> Option<u8> {
        self.src.get(self.pos).copied()
    }

    /// Skips whitespace and `/* */` comments.
    fn skip_ws(&mut self) {
        loop {
            while let Some(b) = self.peek() {
                if b == b' ' || b == b'\t' || b == b'\r' || b == b'\n' {
                    self.pos += 1;
                } else {
                    break;
                }
            }
            if self.peek() == Some(b'/') && self.src.get(self.pos + 1) == Some(&b'*') {
                self.pos += 2;
                while self.pos + 1 < self.src.len() {
                    if self.src[self.pos] == b'*' && self.src[self.pos + 1] == b'/' {
                        self.pos += 2;
                        break;
                    }
                    self.pos += 1;
                }
                if self.pos + 1 >= self.src.len() {
                    self.pos = self.src.len();
                }
                continue;
            }
            break;
        }
    }

    fn read_uint(&mut self) -> Option<u64> {
        let start = self.pos;
        let mut n: u64 = 0;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                n = n.wrapping_mul(10).wrapping_add((b - b'0') as u64);
                self.pos += 1;
            } else {
                break;
            }
        }
        if self.pos == start { None } else { Some(n) }
    }

    /// Reads a `'...'` literal, decoding only the `''` escape. Control
    /// directives such as `\X2\..\X0\` are kept verbatim: entity names are
    /// display-only here and mangling them is harmless.
    fn read_string(&mut self) -> Result<Value, ParseError> {
        debug_assert_eq!(self.peek(), Some(b'\''));
        self.pos += 1;
        let off = self.strings.len() as u32;
        loop {
            let Some(b) = self.peek() else {
                return Err(self.err("unterminated string literal"));
            };
            self.pos += 1;
            if b == b'\'' {
                if self.peek() == Some(b'\'') {
                    self.strings.push(b'\'');
                    self.pos += 1;
                } else {
                    break;
                }
            } else {
                self.strings.push(b);
            }
        }
        Ok(Value::Str {
            off,
            len: self.strings.len() as u32 - off,
        })
    }

    fn read_number(&mut self) -> Result<Value, ParseError> {
        let start = self.pos;
        if matches!(self.peek(), Some(b'+') | Some(b'-')) {
            self.pos += 1;
        }
        let mut is_real = false;
        while let Some(b) = self.peek() {
            match b {
                b'0'..=b'9' => self.pos += 1,
                b'.' => {
                    is_real = true;
                    self.pos += 1;
                }
                b'e' | b'E' => {
                    is_real = true;
                    self.pos += 1;
                    if matches!(self.peek(), Some(b'+') | Some(b'-')) {
                        self.pos += 1;
                    }
                }
                _ => break,
            }
        }
        let text = std::str::from_utf8(&self.src[start..self.pos])
            .map_err(|_| self.err("non-UTF8 in number"))?;
        if is_real {
            // `1.` and `1.E3` are legal STEP reals but not legal Rust literals.
            let v: f64 = text.parse().or_else(|_| {
                let patched = text.replace('.', ".0");
                patched
                    .parse()
                    .map_err(|_| self.err(format!("bad real literal {text:?}")))
            })?;
            Ok(Value::Real(v))
        } else {
            text.parse::<i64>()
                .map(Value::Int)
                .map_err(|_| self.err(format!("bad integer literal {text:?}")))
        }
    }

    fn read_ident(&mut self) -> &'a [u8] {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'-' {
                self.pos += 1;
            } else {
                break;
            }
        }
        &self.src[start..self.pos]
    }

    fn read_value(&mut self) -> Result<Value, ParseError> {
        self.skip_ws();
        let Some(b) = self.peek() else {
            return Err(self.err("unexpected end of file"));
        };
        match b {
            b'#' => {
                self.pos += 1;
                let n = self
                    .read_uint()
                    .ok_or_else(|| self.err("expected digits after '#'"))?;
                Ok(Value::Ref(n as u32))
            }
            b'$' => {
                self.pos += 1;
                Ok(Value::Null)
            }
            b'*' => {
                self.pos += 1;
                Ok(Value::Star)
            }
            b'\'' => self.read_string(),
            b'"' => {
                // Binary literal "0F3A..." -- kept raw; unused by the geometry.
                self.pos += 1;
                let off = self.strings.len() as u32;
                while let Some(c) = self.peek() {
                    self.pos += 1;
                    if c == b'"' {
                        break;
                    }
                    self.strings.push(c);
                }
                Ok(Value::Str {
                    off,
                    len: self.strings.len() as u32 - off,
                })
            }
            b'.' => {
                // Enumerations only; STEP reals always have a digit before '.'.
                self.pos += 1;
                let off = self.strings.len() as u32;
                while let Some(c) = self.peek() {
                    self.pos += 1;
                    if c == b'.' {
                        break;
                    }
                    self.strings.push(c);
                }
                Ok(Value::Enum {
                    off,
                    len: self.strings.len() as u32 - off,
                })
            }
            b'(' => self.read_list(),
            b'0'..=b'9' | b'+' | b'-' => self.read_number(),
            b if b.is_ascii_alphabetic() => {
                // A typed value: `NAME(...)`, as used in complex instances and
                // in SELECT types such as `LENGTH_MEASURE(1.0)`.
                let name = self.read_ident();
                let type_id = self.intern(name);
                self.skip_ws();
                if self.peek() != Some(b'(') {
                    return Err(self.err(format!(
                        "expected '(' after type name {:?}",
                        String::from_utf8_lossy(name)
                    )));
                }
                let inner = self.read_list()?;
                let (start, len) = match inner {
                    Value::List { start, len } => (start, len),
                    _ => unreachable!("read_list always yields a list"),
                };
                let idx = self.typed.len() as u32;
                self.typed.push(TypedEntry {
                    type_id,
                    start,
                    len,
                });
                Ok(Value::Typed(idx))
            }
            other => Err(self.err(format!("unexpected byte {:?}", other as char))),
        }
    }

    /// Reads `( v, v, ... )` into the arena. The elements are gathered on a
    /// shared scratch stack first so that nested lists stay contiguous.
    fn read_list(&mut self) -> Result<Value, ParseError> {
        debug_assert_eq!(self.peek(), Some(b'('));
        self.pos += 1;
        let mark = self.scratch.len();
        self.skip_ws();
        if self.peek() == Some(b')') {
            self.pos += 1;
            return Ok(Value::List { start: 0, len: 0 });
        }
        loop {
            let v = self.read_value()?;
            self.scratch.push(v);
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b')') => {
                    self.pos += 1;
                    break;
                }
                Some(c) => {
                    self.scratch.truncate(mark);
                    return Err(self.err(format!("expected ',' or ')', found {:?}", c as char)));
                }
                None => {
                    self.scratch.truncate(mark);
                    return Err(self.err("unterminated list"));
                }
            }
        }
        let start = self.values.len() as u32;
        let len = (self.scratch.len() - mark) as u32;
        self.values.extend_from_slice(&self.scratch[mark..]);
        self.scratch.truncate(mark);
        Ok(Value::List { start, len })
    }

    /// Advances past `DATA;`. Returns false if the file has no data section.
    fn seek_data_section(&mut self) -> bool {
        // Search for a `DATA` token at statement level. Header strings could in
        // principle contain "DATA;", so require it to be preceded by `;` or the
        // start of a statement and followed by `;`.
        while self.pos < self.src.len() {
            self.skip_ws();
            let start = self.pos;
            if self.peek().is_some_and(|b| b.is_ascii_alphabetic()) {
                let word = self.read_ident();
                self.skip_ws();
                if word.eq_ignore_ascii_case(b"DATA") && self.peek() == Some(b';') {
                    self.pos += 1;
                    return true;
                }
                if self.pos == start {
                    self.pos += 1;
                }
                continue;
            }
            // Skip anything else: strings must be skipped wholesale so their
            // contents are never mistaken for tokens.
            match self.peek() {
                Some(b'\'') => {
                    let _ = self.read_string();
                }
                Some(_) => self.pos += 1,
                None => break,
            }
        }
        false
    }
}

/// Parses the DATA section of a STEP physical file.
pub fn parse(src: &[u8]) -> Result<StepFile, ParseError> {
    // Tolerate a UTF-8 BOM.
    let src = src.strip_prefix(&[0xEF, 0xBB, 0xBF]).unwrap_or(src);
    let mut p = Parser::new(src);

    if !p.seek_data_section() {
        return Err(ParseError {
            offset: 0,
            line: 1,
            message: "no DATA; section found -- is this an ISO-10303-21 file?".into(),
        });
    }

    // Two collections: `pending` holds (id, record) until we know the largest
    // id, then we scatter into a dense table.
    let mut pending: Vec<(u32, Record)> = Vec::with_capacity(src.len() / 90 + 16);
    let mut max_id: u32 = 0;

    loop {
        p.skip_ws();
        let Some(b) = p.peek() else { break };

        if b == b'#' {
            p.pos += 1;
            let id = p
                .read_uint()
                .ok_or_else(|| p.err("expected entity id after '#'"))? as u32;
            p.skip_ws();
            if p.peek() != Some(b'=') {
                return Err(p.err("expected '=' after entity id"));
            }
            p.pos += 1;
            p.skip_ws();

            let rec = match p.peek() {
                // Complex instance: #N=(A(...)B(...));
                Some(b'(') => {
                    p.pos += 1;
                    let mark = p.scratch.len();
                    loop {
                        p.skip_ws();
                        match p.peek() {
                            Some(b')') => {
                                p.pos += 1;
                                break;
                            }
                            Some(b',') => {
                                p.pos += 1;
                            }
                            Some(_) => {
                                let v = p.read_value()?;
                                p.scratch.push(v);
                            }
                            None => {
                                p.scratch.truncate(mark);
                                return Err(p.err("unterminated complex instance"));
                            }
                        }
                    }
                    let start = p.values.len() as u32;
                    let len = (p.scratch.len() - mark) as u32;
                    let vals: Vec<Value> = p.scratch.drain(mark..).collect();
                    p.values.extend_from_slice(&vals);
                    let type_id = p.intern(COMPLEX_TYPE.as_bytes());
                    Record {
                        type_id,
                        start,
                        len,
                    }
                }
                Some(c) if c.is_ascii_alphabetic() => {
                    let name = p.read_ident();
                    let type_id = p.intern(name);
                    p.skip_ws();
                    if p.peek() != Some(b'(') {
                        return Err(p.err("expected '(' after entity type"));
                    }
                    let list = p.read_list()?;
                    let (start, len) = match list {
                        Value::List { start, len } => (start, len),
                        _ => unreachable!(),
                    };
                    Record {
                        type_id,
                        start,
                        len,
                    }
                }
                _ => return Err(p.err("expected entity type or complex instance")),
            };

            p.skip_ws();
            if p.peek() == Some(b';') {
                p.pos += 1;
            }
            max_id = max_id.max(id);
            pending.push((id, rec));
            continue;
        }

        // `ENDSEC;` terminates the data section; anything else at this level is
        // skipped defensively so odd-but-harmless files still load.
        if b.is_ascii_alphabetic() {
            let word = p.read_ident();
            if word.eq_ignore_ascii_case(b"ENDSEC") {
                break;
            }
            p.skip_ws();
            if p.peek() == Some(b';') {
                p.pos += 1;
            }
            continue;
        }
        p.pos += 1;
    }

    let mut records = vec![Record::ABSENT; max_id as usize + 1];
    let entity_count = pending.len();
    for (id, rec) in pending {
        records[id as usize] = rec;
    }

    p.values.shrink_to_fit();
    p.strings.shrink_to_fit();

    Ok(StepFile {
        records,
        values: p.values,
        typed: p.typed,
        strings: p.strings,
        type_names: p.type_names,
        entity_count,
    })
}

// ---------------------------------------------------------------------------
// Small accessors used throughout the model builder.
// ---------------------------------------------------------------------------

impl Value {
    #[inline]
    pub fn as_ref_id(self) -> Option<u32> {
        match self {
            Value::Ref(r) => Some(r),
            _ => None,
        }
    }

    #[inline]
    pub fn as_f64(self) -> Option<f64> {
        match self {
            Value::Real(r) => Some(r),
            Value::Int(i) => Some(i as f64),
            _ => None,
        }
    }

    #[inline]
    pub fn as_i64(self) -> Option<i64> {
        match self {
            Value::Int(i) => Some(i),
            Value::Real(r) => Some(r as i64),
            _ => None,
        }
    }

    #[inline]
    pub fn is_null(self) -> bool {
        matches!(self, Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f(src: &str) -> StepFile {
        parse(src.as_bytes()).expect("parse")
    }

    #[test]
    fn parses_basic_entities() {
        let s = f("ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\n\
             #1=CARTESIAN_POINT('p',(1.,-2.5,3.E1));\n\
             #2=DIRECTION('',(0.,0.,1.));\n\
             ENDSEC;\nEND-ISO-10303-21;\n");
        assert_eq!(s.entity_count, 2);
        assert_eq!(s.type_of(1), "CARTESIAN_POINT");
        let p = s.params(1);
        assert_eq!(s.str_of(p[0]), "p");
        let coords = s.list(p[1]);
        assert_eq!(coords.len(), 3);
        assert_eq!(coords[0].as_f64(), Some(1.0));
        assert_eq!(coords[1].as_f64(), Some(-2.5));
        assert_eq!(coords[2].as_f64(), Some(30.0));
    }

    #[test]
    fn handles_strings_enums_nulls_and_comments() {
        let s = f("DATA;\n/* a comment ; with #99=FAKE(); inside */\n\
             #7=ADVANCED_FACE('it''s',(#1,#2),#3,.T.);\n\
             #8=THING($,*);\nENDSEC;");
        let p = s.params(7);
        assert_eq!(s.str_of(p[0]), "it's");
        assert_eq!(s.list(p[1]).len(), 2);
        assert_eq!(p[2].as_ref_id(), Some(3));
        assert_eq!(s.str_of(p[3]), "T");
        assert!(!s.record(99).exists());
        let q = s.params(8);
        assert!(q[0].is_null());
        assert!(matches!(q[1], Value::Star));
    }

    #[test]
    fn handles_complex_instances() {
        let s = f("DATA;\n#5=(NAMED_UNIT(*)SI_UNIT(.MILLI.,.METRE.));\nENDSEC;");
        assert_eq!(s.type_of(5), COMPLEX_TYPE);
        let si = s.complex_part(5, "SI_UNIT").expect("SI_UNIT part");
        assert_eq!(s.str_of(si[0]), "MILLI");
        assert_eq!(s.str_of(si[1]), "METRE");
        assert!(s.complex_part(5, "NOPE").is_none());
    }

    #[test]
    fn handles_nested_lists_and_typed_values() {
        let s = f("DATA;\n#1=B_SPLINE(((#1,#2),(#3,#4)),LENGTH_MEASURE(2.5));\nENDSEC;");
        let p = s.params(1);
        let outer = s.list(p[0]);
        assert_eq!(outer.len(), 2);
        assert_eq!(s.list(outer[0]).len(), 2);
        assert_eq!(s.list(outer[1])[1].as_ref_id(), Some(4));
        match p[1] {
            Value::Typed(idx) => {
                let t = s.typed_entry(idx);
                assert_eq!(s.type_name(t.type_id), "LENGTH_MEASURE");
                assert_eq!(s.slice(t.start, t.len)[0].as_f64(), Some(2.5));
            }
            other => panic!("expected typed value, got {other:?}"),
        }
    }

    #[test]
    fn expands_short_form_entity_names() {
        // The same faceted face, written in the abbreviated form.
        let s = f("DATA;\n\
             #1=CRTPNT('NONE',(0.,0.,0.));\n\
             #2=CRTPNT('NONE',(1.,0.,0.));\n\
             #3=PLYLP('NONE',(#1,#2));\n\
             #4=FCOTBN('NONE',#3,.T.);\n\
             #5=A2PL3D('NONE',#1,$,$);\n\
             #6=PLANE('NONE',#5);\n\
             #7=FCSRF('NONE',(#4),#6,.T.);\n\
             #8=CLSSHL('NONE',(#7));\n\
             #9=FCTBR('part',#8);\n\
             ENDSEC;");
        assert_eq!(s.type_of(1), "CARTESIAN_POINT");
        assert_eq!(s.type_of(3), "POLY_LOOP");
        assert_eq!(s.type_of(4), "FACE_OUTER_BOUND");
        assert_eq!(s.type_of(5), "AXIS2_PLACEMENT_3D");
        assert_eq!(s.type_of(7), "FACE_SURFACE");
        assert_eq!(s.type_of(8), "CLOSED_SHELL");
        assert_eq!(s.type_of(9), "FACETED_BREP");
        // Names that are already full, or unknown, are left alone.
        assert_eq!(s.type_of(6), "PLANE");
        // Attributes are untouched by the renaming.
        assert_eq!(s.list(s.params(3)[1]).len(), 2);
        assert_eq!(s.params(9)[1].as_ref_id(), Some(8));
    }

    /// An entry mapping a name to itself would make interning recurse forever,
    /// which is a hang rather than a compile error, so check the table itself.
    #[test]
    fn short_name_table_never_maps_a_name_to_itself() {
        for name in [
            "CARTESIAN_POINT",
            "DIRECTION",
            "VECTOR",
            "LINE",
            "CIRCLE",
            "PLANE",
            "ELLIPSE",
            "AXIS2_PLACEMENT_3D",
            "ADVANCED_FACE",
            "CLOSED_SHELL",
            "MANIFOLD_SOLID_BREP",
            "FACETED_BREP",
            "SHAPE_REPRESENTATION",
            "TRIMMED_CURVE",
        ] {
            assert_ne!(
                expand_short_name(name.as_bytes()),
                Some(name),
                "{name} maps to itself"
            );
        }
        // And a file using the full names must still parse.
        let s = f("DATA;\n#1=VECTOR('',#2,1.);\n#2=DIRECTION('',(1.,0.,0.));\nENDSEC;");
        assert_eq!(s.type_of(1), "VECTOR");
        assert_eq!(s.type_of(2), "DIRECTION");
    }

    #[test]
    fn short_and_long_forms_intern_to_the_same_type() {
        let s = f("DATA;\n#1=CRTPNT('',(0.,0.,0.));\n#2=CARTESIAN_POINT('',(1.,1.,1.));\nENDSEC;");
        assert_eq!(s.type_of(1), s.type_of(2));
        assert_eq!(s.record(1).type_id, s.record(2).type_id);
    }

    #[test]
    fn rejects_files_without_data_section() {
        assert!(parse(b"ISO-10303-21;\nHEADER;\nENDSEC;\nEND-ISO-10303-21;").is_err());
    }
}
