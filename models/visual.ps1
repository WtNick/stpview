# Renders the views in views.tsv and compares each against its stored frame,
# allowing zero moved pixels: the renderer is deterministic.
#
#   powershell -File models/visual.ps1            # compare
#   powershell -File models/visual.ps1 -Update    # accept current as reference
#
# baseline.ps1 counts; this looks. Some faults (a misplaced bore, a disc drawn
# across a ring gear) move no count but are obvious in a frame.
#
# The frames are gitignored with the corpus, so this compares a working copy
# with itself: record, change the code, compare.

param([switch]$Update)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot "..\target\release\stpview.exe"
if (-not (Test-Path $exe)) { throw "build first: cargo build --release" }
$views = Join-Path $PSScriptRoot "views.tsv"
$shots = Join-Path $PSScriptRoot "shots"
if (-not (Test-Path $shots)) { New-Item -ItemType Directory $shots | Out-Null }

$rows = Import-Csv -Path $views -Delimiter "`t"
$moved = @()
$missing = @()
$skipped = @()
$checked = 0
foreach ($r in $rows) {
  $model = Join-Path $PSScriptRoot $r.file
  if (-not (Test-Path $model)) { $skipped += $r.file; continue }
  # One reference per view, named for both so a file's views cannot collide.
  $stem = ($r.file -replace '[\\/]', '_') + "_" + $r.name
  $ref = Join-Path $shots "$stem.png"
  $now = Join-Path $shots "$stem.now.png"

  # The args column is a command line, so it has to be split as one.
  $extra = [regex]::Matches($r.args, '[^ ]+') | ForEach-Object { $_.Value }

  # `--thumb` in the args renders through the thumbnail path Explorer uses,
  # which has its own camera, options and import.
  $mode = '--shot'
  if ($extra -contains '--thumb') {
    $mode = '--thumb'
    $extra = $extra | Where-Object { $_ -ne '--thumb' }
  }

  if ($Update) {
    & $exe $model $mode $ref @extra | Out-Null
    if ($LASTEXITCODE -ne 0) { throw "render failed: $($r.file) $($r.name)" }
    continue
  }
  if (-not (Test-Path $ref)) { $missing += "$stem"; continue }

  # Stdout only: in PowerShell 5.1 a redirected native stderr line becomes an
  # error record, which under `Stop` would abort on the first moved view.
  $out = & $exe $model $mode $now @extra --expect $ref | Out-String
  $checked++
  if ($LASTEXITCODE -ne 0) {
    $line = ($out -split "`n" | Where-Object { $_ -match '^visual' } | Select-Object -First 1)
    if (-not $line) { $line = "render failed" }
    $moved += ("{0,-46} {1}" -f $stem, $line.Trim())
  } else {
    # Nothing to look at, so nothing to keep.
    Remove-Item $now -ErrorAction SilentlyContinue
  }
}

if ($Update) {
  "wrote references for $($rows.Count) views to $shots"
  exit 0
}

foreach ($m in $moved) { $m }
if ($missing) { "" ; "no reference yet (run with -Update): $($missing -join ', ')" }
if ($skipped) { "" ; "not in this corpus: $($skipped -join ', ')" }
""
if ($moved) {
  "$($moved.Count) of $checked views moved; the new frames are beside the references as *.now.png"
} else {
  "visual: $checked views, none moved"
}
