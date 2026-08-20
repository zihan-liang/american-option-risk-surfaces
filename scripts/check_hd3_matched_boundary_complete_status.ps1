param(
    [string]$Python = ".\.venv-pinn\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Experiment = Join-Path $RepoRoot "experiments\hd3_matched_boundary_complete_windows_entrypoint.py"

Push-Location $RepoRoot
try {
    if ([System.IO.Path]::IsPathRooted($Python)) {
        $PythonExe = $Python
    }
    else {
        $PythonExe = Join-Path $RepoRoot $Python
    }
    if (-not (Test-Path $PythonExe)) {
        throw "Python interpreter not found: $PythonExe"
    }
    & $PythonExe $Experiment --phase status
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
