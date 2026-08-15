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

function Get-UbuntuDistribution {
    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        $distributions = @(& wsl.exe --list --quiet 2>$null) |
            ForEach-Object { ($_ -replace "`0", "").Trim() } |
            Where-Object { $_ }
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0 -or $distributions.Count -eq 0) {
        throw "No WSL distribution is available. Install WSL2 and Ubuntu first."
    }

    $ubuntu = $distributions | Where-Object { $_ -eq "Ubuntu" } | Select-Object -First 1
    if (-not $ubuntu) {
        $ubuntu = $distributions | Where-Object { $_ -like "Ubuntu*" } | Select-Object -First 1
    }
    if (-not $ubuntu) {
        throw "No Ubuntu WSL distribution was found. Available: $($distributions -join ', ')"
    }
    return $ubuntu
}

$root = [IO.Path]::GetFullPath((Split-Path -Parent $PSScriptRoot))
$configPath = if ([IO.Path]::IsPathRooted($Config)) {
    [IO.Path]::GetFullPath($Config)
} else {
    [IO.Path]::GetFullPath((Join-Path $root $Config))
}
if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
    throw "Configuration file does not exist: $configPath"
}
$hasContinuationOverride = (
    $ResumeStaggerMaxIterations -gt 0 -or
    $ResumeMaximumSubdivisions -gt 0 -or
    $ResumeMinimumIncrement -gt 0.0
)
if ($hasContinuationOverride -and -not $Resume) {
    throw "Resume continuation controls require -Resume."
}

$distribution = Get-UbuntuDistribution
$windowsRootPayload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($root))
$windowsConfigPayload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($configPath))

$linuxScript = @'
set -o pipefail

conda_sh="$HOME/miniconda3/etc/profile.d/conda.sh"
if [[ ! -f "$conda_sh" ]]; then
    echo "Conda activation script not found: $conda_sh" >&2
    exit 2
fi
source "$conda_sh" || exit 2
conda activate base >/dev/null || exit 2
set -eu

for command_name in python gcc; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "$command_name is unavailable after 'conda activate base'." >&2
        exit 2
    fi
done

GRAPHFRACTURE_WINDOWS_ROOT="$(
    printf '%s' "$GRAPHFRACTURE_WINDOWS_ROOT_B64" | base64 -d
)"
GRAPHFRACTURE_WINDOWS_CONFIG="$(
    printf '%s' "$GRAPHFRACTURE_WINDOWS_CONFIG_B64" | base64 -d
)"
GRAPHFRACTURE_WSL_ROOT="$(wslpath -a -u "$GRAPHFRACTURE_WINDOWS_ROOT")"
GRAPHFRACTURE_WSL_CONFIG="$(wslpath -a -u "$GRAPHFRACTURE_WINDOWS_CONFIG")"
cd "$GRAPHFRACTURE_WSL_ROOT"
export PYTHONPATH="$GRAPHFRACTURE_WSL_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1

resume_args=()
if [[ "$GRAPHFRACTURE_RESUME" == "1" ]]; then
    resume_args+=(--resume)
fi
if (( GRAPHFRACTURE_RESUME_STAGGER_MAX_ITERATIONS > 0 )); then
    resume_args+=(
        --resume-stagger-max-iterations
        "$GRAPHFRACTURE_RESUME_STAGGER_MAX_ITERATIONS"
    )
fi
if (( GRAPHFRACTURE_RESUME_MAXIMUM_SUBDIVISIONS > 0 )); then
    resume_args+=(
        --resume-maximum-subdivisions
        "$GRAPHFRACTURE_RESUME_MAXIMUM_SUBDIVISIONS"
    )
fi
if [[ "$GRAPHFRACTURE_RESUME_MINIMUM_INCREMENT" != "0" ]]; then
    resume_args+=(
        --resume-minimum-increment
        "$GRAPHFRACTURE_RESUME_MINIMUM_INCREMENT"
    )
fi

if (( GRAPHFRACTURE_MPI_RANKS > 1 )); then
    if ! command -v mpiexec >/dev/null 2>&1; then
        echo "mpiexec is unavailable after 'conda activate base'." >&2
        exit 2
    fi
    # PETSc/MUMPS supplies the distributed parallelism.  Prevent every MPI
    # rank from independently spawning a full BLAS/OpenMP thread pool.
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        mpiexec -n "$GRAPHFRACTURE_MPI_RANKS" \
            python -m graphfracture "$GRAPHFRACTURE_WSL_CONFIG" "${resume_args[@]}"
else
    python -m graphfracture "$GRAPHFRACTURE_WSL_CONFIG" "${resume_args[@]}"
fi
'@

$payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($linuxScript))
$previousWslEnv = $env:WSLENV
$previousRootPayload = $env:GRAPHFRACTURE_WINDOWS_ROOT_B64
$previousConfigPayload = $env:GRAPHFRACTURE_WINDOWS_CONFIG_B64
$previousRanks = $env:GRAPHFRACTURE_MPI_RANKS
$previousResume = $env:GRAPHFRACTURE_RESUME
$previousResumeIterations = $env:GRAPHFRACTURE_RESUME_STAGGER_MAX_ITERATIONS
$previousResumeSubdivisions = $env:GRAPHFRACTURE_RESUME_MAXIMUM_SUBDIVISIONS
$previousResumeIncrement = $env:GRAPHFRACTURE_RESUME_MINIMUM_INCREMENT
try {
    $env:GRAPHFRACTURE_WINDOWS_ROOT_B64 = $windowsRootPayload
    $env:GRAPHFRACTURE_WINDOWS_CONFIG_B64 = $windowsConfigPayload
    $env:GRAPHFRACTURE_MPI_RANKS = [string]$MpiRanks
    $env:GRAPHFRACTURE_RESUME = if ($Resume) { "1" } else { "0" }
    $env:GRAPHFRACTURE_RESUME_STAGGER_MAX_ITERATIONS = [string]$ResumeStaggerMaxIterations
    $env:GRAPHFRACTURE_RESUME_MAXIMUM_SUBDIVISIONS = [string]$ResumeMaximumSubdivisions
    $env:GRAPHFRACTURE_RESUME_MINIMUM_INCREMENT = (
        $ResumeMinimumIncrement.ToString("R", [Globalization.CultureInfo]::InvariantCulture)
    )
    $forwarded = @(
        "GRAPHFRACTURE_WINDOWS_ROOT_B64",
        "GRAPHFRACTURE_WINDOWS_CONFIG_B64",
        "GRAPHFRACTURE_MPI_RANKS",
        "GRAPHFRACTURE_RESUME",
        "GRAPHFRACTURE_RESUME_STAGGER_MAX_ITERATIONS",
        "GRAPHFRACTURE_RESUME_MAXIMUM_SUBDIVISIONS",
        "GRAPHFRACTURE_RESUME_MINIMUM_INCREMENT"
    )
    $env:WSLENV = (@($previousWslEnv, ($forwarded -join ":")) | Where-Object { $_ }) -join ":"

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & wsl.exe -d $distribution -- bash -lc "echo $payload | base64 -d | bash"
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "The WSL DOLFINx run failed with exit code $exitCode."
    }
} finally {
    $env:WSLENV = $previousWslEnv
    $env:GRAPHFRACTURE_WINDOWS_ROOT_B64 = $previousRootPayload
    $env:GRAPHFRACTURE_WINDOWS_CONFIG_B64 = $previousConfigPayload
    $env:GRAPHFRACTURE_MPI_RANKS = $previousRanks
    $env:GRAPHFRACTURE_RESUME = $previousResume
    $env:GRAPHFRACTURE_RESUME_STAGGER_MAX_ITERATIONS = $previousResumeIterations
    $env:GRAPHFRACTURE_RESUME_MAXIMUM_SUBDIVISIONS = $previousResumeSubdivisions
    $env:GRAPHFRACTURE_RESUME_MINIMUM_INCREMENT = $previousResumeIncrement
}
