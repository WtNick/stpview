# Downloads the public part of the test corpus from steptools.com.
#
#   powershell -File models/fetch.ps1

$ErrorActionPreference = 'Continue'
$base = "https://www.steptools.com/docs/stpfiles"
$root = $PSScriptRoot

# schema, filename, what it exercises
$files = @(
  @('ap203', 'block.stp',               'trivially simple - smoke test'),
  @('ap203', 'angle1.stp',              'small ACIS part'),
  @('ap203', 'cubcylso.stp',            'advanced brep only'),
  @('ap203', 'cubsomcy.stp',            'two brep representations in one file'),
  @('ap203', 'cylcub.stp',              'FACETED_BREP with POLY_LOOP'),
  @('ap203', 'gear.stp',                'geometrically bounded surfaces (CURVE_BOUNDED_SURFACE)'),
  @('ap203', '4pinplug.stp',            'small ACIS part'),
  @('ap203', 'clevis21.stp',            'INTERSECTION_CURVE edges'),
  @('ap203', 'doghouse.stp',            'INTERSECTION_CURVE edges'),
  @('ap203', 'as1_pe.stp',              'context dependent assembly structure'),
  @('ap203', 'bull.stp',                'CADDS 5 translator, QUASI_UNIFORM_CURVE'),
  @('ap203', 'bernetl.stp',             'faceted plus class 2 geometry'),
  @('ap203', 'piston.stp',              'ORIENTED_FACE, INTERSECTION_CURVE, NURBS'),
  @('ap203', 'socks.stp',               'Part 21 short-form entity names'),
  @('ap203', 'moon_buggy.stp',          'single brep with bspline surfaces'),
  @('ap203', 'moon_buggy_asm.stp',      'MAPPED_ITEM assembly placement'),
  @('ap203', 'nasty_cheese.stp',        'pathological geometry'),
  @('ap203', 'turbine.stp',             'Unigraphics, NURBS heavy'),
  @('ap203', 'rear.stp',                'large Pro/E model, SURFACE_OF_REVOLUTION'),
  # The same AS1 assembly written by five CAD systems: a differential test for
  # translator quirks against identical nominal geometry.
  @('ap214', 'as1-ac-214.stp',          'AS1 assembly via AutoCAD'),
  @('ap214', 'as1-ec-214.stp',          'AS1 assembly via Euclid, SURFACE_CURVE'),
  @('ap214', 'as1-md-214.stp',          'AS1 assembly via MicroStation'),
  @('ap214', 'as1-tc-214.stp',          'AS1 assembly via Theorem'),
  @('ap214', 'as1-ug-214.stp',          'AS1 assembly via Unigraphics'),
  @('ap214', 'io1-ug-214.stp',          'AP214 single part'),
  @('ap214', 'boxy_with_flatness.stp',  'AP214 with PMI annotation'),
  @('ap214', 'boxy_with_linearsize.stp','AP214 with PMI dimensions')
)

foreach ($schema in ($files | ForEach-Object { $_[0] } | Sort-Object -Unique)) {
  New-Item -ItemType Directory -Force (Join-Path $root $schema) | Out-Null
}

$ok = 0; $bad = 0
foreach ($f in $files) {
  $schema, $name, $why = $f
  $dest = Join-Path $root "$schema\$name"
  & curl.exe --fail --silent --show-error --location --max-time 120 -o $dest "$base/$schema/$name"
  if ($LASTEXITCODE -ne 0 -or -not (Test-Path $dest)) {
    "FAIL      $schema/$name"; $bad++; continue
  }
  # A 404 page would download happily, so require a real Part 21 header.
  if ((Get-Content $dest -TotalCount 1 -ErrorAction SilentlyContinue) -notmatch 'ISO-10303-21') {
    "NOT STEP  $schema/$name"; Remove-Item $dest -Force; $bad++; continue
  }
  "{0,-9} {1,-26} {2,6} KB  {3}" -f $schema, $name, [int]((Get-Item $dest).Length / 1KB), $why
  $ok++
}
""
"downloaded $ok, failed $bad"
