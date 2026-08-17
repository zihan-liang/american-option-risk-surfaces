param([string]$Python = '.\.venv-pinn\Scripts\python.exe')
$ErrorActionPreference='Stop'
$RepoRoot=(Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot
if (![System.IO.Path]::IsPathRooted($Python)) { $Python = Join-Path $RepoRoot $Python }
$Experiment=Join-Path $RepoRoot 'experiments\hd1_windows_high_dimensional_pinn_48h.py'
if (!(Test-Path $Python)) { Write-Error "Python not found: $Python"; exit 2 }
# Strictly read-only: the Python status phase reads config/manifests/status only.
& $Python $Experiment --phase status
exit $LASTEXITCODE
