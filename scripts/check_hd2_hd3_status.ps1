param(
    [string]$Python = ".\.venv-pinn\Scripts\python.exe"
)

$Runner = Join-Path $PSScriptRoot "run_hd2_hd3_windows.ps1"
& $Runner -Phase status -Python $Python
exit $LASTEXITCODE
