# Compares `stpview --stats --check` over the corpus against baseline.tsv.
#
#   powershell -File models/baseline.ps1            # compare
#   powershell -File models/baseline.ps1 -Update    # accept current as baseline
#
# Every difference is either an improvement to record or a change to explain.

param([switch]$Update)

$ErrorActionPreference = 'Stop'
$exe = Join-Path $PSScriptRoot "..\target\release\stpview.exe"
if (-not (Test-Path $exe)) { throw "build first: cargo build --release" }
$path = Join-Path $PSScriptRoot "baseline.tsv"

$models = Get-ChildItem $PSScriptRoot -Recurse -Include *.stp, *.step -File | Sort-Object FullName
$now = @{}
foreach ($f in $models) {
  $out = & $exe --stats --check $f.FullName 2>&1 | Out-String
  $ok = $LASTEXITCODE -eq 0
  $rel = $f.FullName.Substring($PSScriptRoot.Length).TrimStart('\') -replace '\\', '/'
  $solids = if ($out -match 'build\s+(\d+) solids') { [int]$Matches[1] } else { 0 }
  $tris = if ($out -match 'tess\s+(\d+) triangles') { [int]$Matches[1] } else { 0 }
  $bad = if ($out -match '(\d+) of (\d+) faces not tessellated') { [int]$Matches[1] } else { 0 }
  # A body with no faces is a loss nothing else reports.
  $lost = if ($out -match 'WARNING (\d+) bodies had no faces') { [int]$Matches[1] } else { 0 }
  # Cracks between faces, which a render rarely shows.
  $crack = if ($out -match 'check\s+(\d+) of \d+ mesh edges unpaired') { [int]$Matches[1] } else { 0 }
  $dup = if ($out -match 'unpaired, (\d+) over-shared') { [int]$Matches[1] } else { 0 }
  $sharp = if ($out -match 'edges\s+(\d+) sharp') { [int]$Matches[1] } else { 0 }
  $tan = if ($out -match 'sharp, (\d+) tangent') { [int]$Matches[1] } else { 0 }
  # Mesh memory, which catches a vertex format that grew by accident.
  $mb = if ($out -match 'memory\s+(\d+) MB') { [int]$Matches[1] } else { 0 }
  # Distinct components among the solids: tessellation is shared between
  # repeats, so losing a repeat costs time without moving any other column.
  $uniq = if ($out -match 'dedup\s+(\d+) unique') { [int]$Matches[1] } else { $solids }
  $now[$rel] = [pscustomobject]@{
    File = $rel; Loads = [int]$ok; Solids = $solids; Unique = $uniq; Tris = $tris; Bad = $bad
    Lost = $lost; Cracks = $crack; Dup = $dup; Sharp = $sharp; Tangent = $tan
    MB = $mb
  }
}

if ($Update) {
  $now.Values | Sort-Object File |
    Export-Csv -Path $path -Delimiter "`t" -NoTypeInformation -Encoding utf8
  "wrote $($now.Count) rows to $path"
  exit 0
}
if (-not (Test-Path $path)) { throw "no baseline; run with -Update first" }

$was = @{}
foreach ($r in (Import-Csv -Path $path -Delimiter "`t")) { $was[$r.File] = $r }

# Which direction is better for each column.
$lowerIsBetter = @{
  Bad = $true; Lost = $true; Cracks = $true; Dup = $true; MB = $true
  Tris = $false; Solids = $false; Loads = $false
}
$diffs = @()
$skipped = @()
foreach ($k in ($now.Keys + $was.Keys | Sort-Object -Unique)) {
  if (-not $was.ContainsKey($k)) { $diffs += "NEW      $k"; continue }
  # Not on disk; the corpus is downloaded, not committed.
  if (-not $now.ContainsKey($k)) { $skipped += $k; continue }
  foreach ($col in 'Loads', 'Solids', 'Unique', 'Tris', 'Bad', 'Lost', 'Cracks', 'Dup', 'Sharp', 'Tangent', 'MB') {
    $a = [int]$was[$k].$col
    $b = [int]$now[$k].$col
    if ($a -eq $b) { continue }
    $verdict = if ($lowerIsBetter[$col] -ne $null -and $lowerIsBetter[$col]) {
      if ($b -lt $a) { 'better' } else { 'WORSE' }
    } elseif ($col -eq 'Loads') {
      if ($b -gt $a) { 'better' } else { 'WORSE' }
    } else { 'changed' }
    $diffs += ("{0,-8} {1,-46} {2} {3} -> {4}" -f $verdict, $k, $col, $a, $b)
  }
}

if (-not $diffs) {
  "baseline: $($now.Count) files, no change"
} else {
  $diffs
  ""
  "$(($diffs | Where-Object { $_ -match 'WORSE' }).Count) regressions, " +
  "$(($diffs | Where-Object { $_ -match 'better' }).Count) improvements, " +
  "$(($diffs | Where-Object { $_ -match 'changed|NEW' }).Count) to explain"
}
if ($skipped) { ""; "not in this corpus: $($skipped -join ', ')" }
