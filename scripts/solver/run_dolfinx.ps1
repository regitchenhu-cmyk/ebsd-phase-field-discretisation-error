param(
    [string]$Config = "examples/sent_graph_hydrogen_onset.toml",
    [ValidateRange(1, [int]::MaxValue)]
    [int]$MpiRanks = 1,
    [switch]$Resume,
    [ValidateRange(0, [int]::MaxValue)]
    [int]$ResumeStaggerMaxIterations = 0,
    [ValidateRange(0, [int]::MaxValue)]
    [int]$ResumeMaximumSubdivisions = 0,
    [ValidateRange(0.0, [double]::MaxValue)]
    [double]$ResumeMinimumIncrement = 0.0
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$image = "dolfinx/dolfinx:v0.11.0@sha256:2ae4bfbc0d9077268880faf04c72750528bee986c94ab223a2c159969bd56fa8"
$dockerArgs = @(
    "run", "--rm",
    "-v", "${root}:/workspace",
    "-w", "/workspace",
    "-e", "PYTHONPATH=/workspace/src",
    "-e", "OMPI_ALLOW_RUN_AS_ROOT=1",
    "-e", "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1",
    $image
)

$runArgs = @("python3", "-m", "graphfracture", $Config)
if ($Resume) {
    $runArgs += "--resume"
}
$hasContinuationOverride = (
    $ResumeStaggerMaxIterations -gt 0 -or
    $ResumeMaximumSubdivisions -gt 0 -or
    $ResumeMinimumIncrement -gt 0.0
)
if ($hasContinuationOverride -and -not $Resume) {
    throw "Resume continuation controls require -Resume."
}
if ($ResumeStaggerMaxIterations -gt 0) {
    $runArgs += @("--resume-stagger-max-iterations", $ResumeStaggerMaxIterations)
}
if ($ResumeMaximumSubdivisions -gt 0) {
    $runArgs += @("--resume-maximum-subdivisions", $ResumeMaximumSubdivisions)
}
if ($ResumeMinimumIncrement -gt 0.0) {
    $runArgs += @(
        "--resume-minimum-increment",
        $ResumeMinimumIncrement.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
    )
}
if ($MpiRanks -gt 1) {
    $dockerArgs += @("mpiexec", "-n", $MpiRanks) + $runArgs
} else {
    $dockerArgs += $runArgs
}

& docker @dockerArgs
