param(
    [string]$Python = ".\.venv-pinn\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
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
    & $PythonExe "experiments\hd3_matched_american_put_comparison.py" --phase status
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
