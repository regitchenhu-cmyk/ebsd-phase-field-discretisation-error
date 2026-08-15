param()

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
$distribution = Get-UbuntuDistribution
$windowsRootPayload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($root))

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

for command_name in python gcc mpiexec; do
    if ! command -v "$command_name" >/dev/null 2>&1; then
        echo "$command_name is unavailable after 'conda activate base'." >&2
        exit 2
    fi
done
if ! python -c 'import pytest' >/dev/null 2>&1; then
    echo "pytest is missing from the Conda base environment." >&2
    echo "Install it with: python -m pip install 'pytest>=8,<9'" >&2
    exit 2
fi

GRAPHFRACTURE_WINDOWS_ROOT="$(
    printf '%s' "$GRAPHFRACTURE_WINDOWS_ROOT_B64" | base64 -d
)"
GRAPHFRACTURE_WSL_ROOT="$(wslpath -a -u "$GRAPHFRACTURE_WINDOWS_ROOT")"
cd "$GRAPHFRACTURE_WSL_ROOT"
export PYTHONPATH="$GRAPHFRACTURE_WSL_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export GRAPHFRACTURE_REQUIRE_DOLFINX=1

# A fresh Python process per file is intentional for both serial and MPI
# gates.  Reusing one process across every DOLFINx integration file can retain
# PETSc/JIT native state and make the later Real-element path-control test fail
# or spin even though every file passes in a clean process.
dolfinx_test_files=(
    tests/test_dolfinx_smoke.py
    tests/test_dolfinx_verification.py
    tests/test_path_control_dolfinx.py
    tests/test_hybrid_runner_dolfinx.py
    tests/test_hybrid_restart_dolfinx.py
)
for test_file in "${dolfinx_test_files[@]}"; do
    python -m pytest -q -p no:cacheprovider "$test_file"
done
for test_file in "${dolfinx_test_files[@]}"; do
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
        mpiexec -n 2 python -m pytest -q -p no:cacheprovider "$test_file"
done
'@

$payload = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($linuxScript))
$previousWslEnv = $env:WSLENV
$previousRootPayload = $env:GRAPHFRACTURE_WINDOWS_ROOT_B64
try {
    $env:GRAPHFRACTURE_WINDOWS_ROOT_B64 = $windowsRootPayload
    $env:WSLENV = (
        @($previousWslEnv, "GRAPHFRACTURE_WINDOWS_ROOT_B64") |
            Where-Object { $_ }
    ) -join ":"

    $previousPreference = $ErrorActionPreference
    try {
        $ErrorActionPreference = "Continue"
        & wsl.exe -d $distribution -- bash -lc "echo $payload | base64 -d | bash"
        $exitCode = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $previousPreference
    }
    if ($exitCode -ne 0) {
        throw "The WSL DOLFINx serial/2-rank verification failed with exit code $exitCode."
    }
} finally {
    $env:WSLENV = $previousWslEnv
    $env:GRAPHFRACTURE_WINDOWS_ROOT_B64 = $previousRootPayload
}
