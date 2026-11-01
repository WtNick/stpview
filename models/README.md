# Test corpus

Regression models for `baseline.ps1` and `visual.ps1`: 27 small samples from
[steptools.com](https://www.steptools.com/docs/stpfiles/), each chosen for a
feature (listed in `fetch.ps1`). They are third-party data, so they are not
committed; fetch them with:

```bash
powershell -File models/fetch.ps1
```

`fetch.ps1` rejects any download that does not start with `ISO-10303-21`. The
scripts also pick up any other `.stp` or `.step` dropped under this folder, and
skip rows for models that are not present.

The five `as1-*-214.stp` files are the same assembly exported by five CAD
systems. A difference between their results points at a translator quirk the
reader mishandles, not at the geometry.

## Scripts

All of them run `target/release/stpview.exe`, so run `cargo build --release`
first.

| Script | |
| --- | --- |
| `sweep.ps1` | `--stats` over every model: a summary table plus every diagnostic the reader printed |
| `baseline.ps1` | `--stats --check` against `baseline.tsv`; each difference is labelled better, WORSE or changed. `-Update` accepts the current numbers. |
| `visual.ps1` | Renders the views in `views.tsv` and compares them to stored frames in `shots/`, with zero pixels allowed to move. `-Update` re-records. |

The stored frames are gitignored, so `visual.ps1` compares a working copy with
itself. Record the frames before a change, then compare after it. Re-record
after a driver or GPU change, and look at what you are recording.

`composite.toml` is a scene file used by one of the views. It covers every
scene-file feature, including mixed units.
