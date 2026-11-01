# Contributing

## Checks

```bash
cargo build --release                  # the scripts run target/release/stpview.exe
cargo test --release                   # does not build the exe; build first
powershell -File models/fetch.ps1      # once: download the sample corpus
powershell -File models/baseline.ps1   # corpus statistics against baseline.tsv
powershell -File models/visual.ps1     # stored views, zero moved pixels
```

After a reader or tessellator change, run all three. Triangle counts moving is
normal, but explain each change before accepting it with `-Update`. See
[models/README.md](models/README.md).

## Conventions

- Comments explain *why* the code is as it is now. They do not tell its
  history.
- Source files are ASCII only. Check with
  `grep -rnP '[^\x00-\x7F]' crates/*/src/`.
- Diagnostics are part of the product. Anything dropped is counted in
  `--stats` and named.
- Look at the render (`--shot`, `--view`, `--wire`, `--edges all`,
  `--ao off`) before believing a statistic. For brightness, compare numbers
  with `--no-ground` rather than judging by eye.
- Prefer property tests (area against a closed form,
  `project(eval(uv)) == uv`) to golden values.
- A new derived quantity gets a `baseline.ps1` column.
- A scene's world is millimetres; a single file's world is its own units.

## Where STEP bugs hide

1. `STYLED_ITEM` often styles faces, not bodies. A face's style beats its
   body's; colour and opacity inherit separately.
2. Derived attributes are written `*` and shift every attribute after them.
   Find references by scanning attributes, not by fixed index.
3. Closedness is never stated. Ask the geometry whether a curve's ends meet.
4. "No boundary" has three spellings: an empty bound list, a seam traced out
   and back, and a `VERTEX_LOOP`.
5. Features below the tolerance must still exist; a tiny arc must not be
   chorded away.
6. Degeneracy tests scale with what they test. Exact zero comes back as
   `1e-16`.
7. Angles carry the file's unit, usually degrees.
8. Never separate a point from its parameters. Shift, reverse and join `uv`
   and `pts` together, and only add whole periods to a parameter.
9. Each bound is projected alone, so bring holes into their ring's period.
10. A vertex is not always where its edges meet, nor the extent of its
    edges.
11. A triangulator's success is a claim. Check every result for its whole
    outline.

## Renderer invariants

Each of these is guarded by a test or a stored view.

- Depth is reversed (near = 1, far = 0, `GreaterEqual`, clear to 0). The
  shadow map is orthographic and not reversed.
- The near plane comes from the eye's distance to the model, not from the
  orbit target.
- The section cap is a stencil parity count of the geometry on the far side
  of the plane. Sheets are never capped.
- Every shader that declares the uniform block declares all of it.
- Paint is blended as `c + (p - c) * a`, so zero paint leaves the frame
  unchanged to the byte.
- Every pass, including the shadow map and the occlusion prepass, agrees on
  what is hidden.
- Camera roll is kept exactly zero unless the camera is rolled.
- A still view averages 16 jittered frames. Frame 0 is unshifted.
- Streaming and refinement share one tessellation path. Refinement tolerance
  depends on position, so neighbouring faces sample shared edges identically.
- A vertex is 20 bytes.
- The thumbnail picture lives in `stpview::thumb::render`, so `--thumb`
  draws exactly what Explorer does.

## Debug switches

| Variable | |
| --- | --- |
| `STPVIEW_PROBE="x,y,z[,r]"` | Dump every face whose outline passes near a point. |
| `STPVIEW_CDT=1\|2` | Outlines the constrained triangulator refused or returned incomplete. `2` prints small ones in full. |
| `STPVIEW_MESH_EDGES=1` | Draw triangles instead of B-rep edges in the overlay. |
| `STPVIEW_TIMING=1` | Split the renderer's start-up cost. |
| `STPVIEW_STREAM=1` | Make `--shot` build buffers the streaming way. |
| `STPVIEW_THUMB_LOG=<file>` | Log thumbnail failures from inside Explorer. Set it with `setx`. |
