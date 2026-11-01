# Runs every model under this folder through `stpview --stats` and prints a
# summary table plus every diagnostic the reader printed.
#
#   powershell -File models/sweep.ps1

$exe = Join-Path $PSScriptRoot "..\target\release\stpview.exe"
if (-not (Test-Path $exe)) { throw "build first: cargo build --release" }

$rows = @()
$notes = @()

# Both extensions are in common use, and Windows matching is case-insensitive.
$models = Get-ChildItem $PSScriptRoot -Recurse -Include *.stp, *.step -File | Sort-Object FullName
foreach ($f in $models) {
  $out = & $exe --stats $f.FullName 2>&1 | Out-String
  $code = $LASTEXITCODE
  $rel = $f.FullName.Substring($PSScriptRoot.Length).TrimStart('\')

  $solids = 0; $faces = 0
  if ($out -match 'build\s+(\d+) solids, (\d+) faces') { $solids = [int]$Matches[1]; $faces = [int]$Matches[2] }
  $ents = if ($out -match 'parse\s+(\d+) entities') { [int]$Matches[1] } else { 0 }
  $tris = if ($out -match 'tess\s+(\d+) triangles') { [int]$Matches[1] } else { 0 }
  $bad = if ($out -match '(\d+) of (\d+) faces not tessellated') { [int]$Matches[1] } else { 0 }
  $sharp = if ($out -match 'edges\s+(\d+) sharp') { [int]$Matches[1] } else { 0 }

  # `--stats` ends with a per-solid listing whose rows are also indented, so
  # the remark scan below stops where that begins.
  $split = ($out -split "`n")
  $solidsHeader = [Array]::IndexOf(($split | ForEach-Object { $_.Trim() }), 'solids:')
  if ($solidsHeader -lt 0) { $solidsHeader = $split.Count }

  $status = if ($code -eq 0) { 'ok' } else {
    $err = ($out -split "`n" | Where-Object { $_ -match '^error:' } | Select-Object -First 1)
    if ($err) { $err.Trim() -replace '^error:\s*', '' } else { "exit $code" }
  }

  # Every indented remark except known summaries. Filtering out rather than in
  # means a newly added diagnostic shows up without registering it here. The
  # per-item detail under a WARNING is dropped too; the WARNING carries the
  # count.
  $routine = @(
    '^bounds ',                       # informational extents
    '^\d+ of \d+ solids moved by',    # assembly placement summary
    '^\d+ seam bands, \d+ apex',      # closure summary
    '(covered twice|left uncovered) at ',   # detail under the wrong-area WARNING
    '^(crack|doubled) near '                # detail under the --check counts
  )
  foreach ($line in (($out -split "`n") | Select-Object -First $solidsHeader)) {
    $t = $line.Trim()
    if ($line -notmatch '^\s{4,}\S' -or -not $t) { continue }
    if ($routine | Where-Object { $t -match $_ }) { continue }
    $notes += "$rel : $t"
  }

  $rows += [pscustomobject]@{
    File = $rel; Entities = $ents; Solids = $solids; Faces = $faces
    Tris = $tris; Sharp = $sharp; Bad = $bad; Status = $status
  }
}

$rows | Format-Table -AutoSize | Out-String -Width 200
"--- flags ---"
$notes | Sort-Object -Unique
""
"$(($rows | Where-Object { $_.Status -eq 'ok' }).Count) of $($rows.Count) loaded"
"$(($rows | Where-Object { $_.Status -eq 'ok' -and $_.Bad -eq 0 }).Count) tessellated with zero failed faces"
